#!/usr/bin/env node
// Capture a scan-studio report.html as PNG screenshots and/or a turntable MP4 of its 3D view, in headless Chrome.
// usage: scan-studio render <report.html> [--png out.png] [--full-png out.png] [--mode after|before|compare]
//        [--mp4 out.mp4 --seconds 8 --fps 30] [--width 1440] [--height 900]
import { createRequire } from 'node:module'
import { execFileSync, spawn } from 'node:child_process'
import { existsSync, realpathSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const ARGS = ['--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--use-angle=swiftshader-webgl', '--enable-webgl']
const NUMERIC = new Set(['width', 'height', 'seconds', 'fps'])
const KNOWN = new Set(['png', 'fullPng', 'mode', 'mp4', 'seconds', 'fps', 'width', 'height'])

function opts (argv) {
  const o = new Map([['width', 1440], ['height', 900], ['seconds', 8], ['fps', 30], ['mode', 'after']])
  const pos = []
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (!a.startsWith('--')) { pos.push(a); continue }
    const key = a.slice(2).replace(/-(.)/g, (_, c) => c.toUpperCase())
    if (!KNOWN.has(key)) throw new Error(`unknown option ${a}`)
    const v = argv[++i]
    if (v === undefined) throw new Error(`${a} needs a value`)
    o.set(key, NUMERIC.has(key) ? Number(v) : v)
  }
  if (!pos[0]) throw new Error('usage: render <report.html> [--png out.png] [--full-png out.png] [--mp4 out.mp4]')
  for (const k of NUMERIC) if (!(o.get(k) > 0 && o.get(k) <= 4000)) throw new Error(`--${k} must be a positive number`)
  if (!['after', 'before', 'compare'].includes(o.get('mode'))) throw new Error('--mode is after, before or compare')
  return { o, html: resolve(pos[0]) }
}

function playwright () {
  // resolve from this file's folder or the global npm root, never from the current working folder
  const require = createRequire(import.meta.url)
  let g = ''
  try { g = execFileSync('npm', ['root', '-g'], { encoding: 'utf8', cwd: ROOT }).trim() } catch {}
  const cands = [process.env.PLAYWRIGHT_CORE, 'playwright-core', g && join(g, 'playwright-core'),
    g && join(g, '@playwright/cli/node_modules/playwright-core'), g && join(g, '@playwright/test/node_modules/playwright-core')]
  for (const c of cands.filter(Boolean)) { try { return require(c).chromium } catch {} }
  throw new Error('Playwright not found: npm i -g playwright-core (or set PLAYWRIGHT_CORE)')
}

function ffmpeg () {
  if (process.env.FF) return process.env.FF
  try { return execFileSync('which', ['ffmpeg'], { encoding: 'utf8' }).trim() } catch {}
  const py = process.env.SCAN_STUDIO_PYTHON || (existsSync(join(ROOT, '.venv/bin/python')) ? join(ROOT, '.venv/bin/python') : 'python3')
  try { return execFileSync(py, ['-I', '-c', 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'], { encoding: 'utf8', cwd: '/' }).trim() } catch {}
  throw new Error('ffmpeg not found: install ffmpeg, or pip install imageio-ffmpeg, or set FF')
}

async function launch (chromium) {
  const tries = [process.env.SCAN_STUDIO_CHROME ? { executablePath: process.env.SCAN_STUDIO_CHROME } : null, { channel: 'chrome' }, {}].filter(Boolean)
  let last
  for (const t of tries) { try { return await chromium.launch({ ...t, args: ARGS }) } catch (e) { last = e } }
  throw new Error(`no browser could start: ${String(last && last.message).split('\n')[0]}`)
}

async function main () {
  const { o, html } = opts(process.argv.slice(2))
  if (!existsSync(html)) throw new Error('no such file: ' + html)
  const browser = await launch(playwright())
  try {
    const page = await browser.newPage({ viewport: { width: o.get('width'), height: o.get('height') }, deviceScaleFactor: 1 })
    await page.route('**/*', (route) => (route.request().url().startsWith('file:') ? route.continue() : route.abort()))
    const errors = []
    page.on('pageerror', (e) => errors.push(e.message))
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
    await page.goto(pathToFileURL(realpathSync(html)).href + '?mode=' + o.get('mode'))
    await page.waitForFunction(() => window.__scanReady && window.__scanReady(), null, { timeout: 60000 })
    await page.evaluate((m) => { window.__scanMode(m); window.__scanAngle(0.6) }, o.get('mode'))
    await page.waitForTimeout(500)
    if (o.get('png')) await page.locator('#stage').screenshot({ path: resolve(o.get('png')) })
    if (o.get('fullPng')) await page.screenshot({ path: resolve(o.get('fullPng')), fullPage: true })
    if (o.get('mp4')) {
      const n = Math.round(o.get('seconds') * o.get('fps'))
      const ff = spawn(ffmpeg(), ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(o.get('fps')), '-i', 'pipe:0',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', '-movflags', '+faststart', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', resolve(o.get('mp4'))],
      { stdio: ['pipe', 'inherit', 'inherit'] })
      for (let i = 0; i < n; i++) {
        await page.evaluate((a) => window.__scanAngle(a), 0.6 + (i / n) * Math.PI * 2)
        const buf = await page.locator('#stage').screenshot({ type: 'jpeg', quality: 92 })
        if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r))
      }
      ff.stdin.end()
      const code = await new Promise((r) => ff.on('close', r))
      if (code !== 0) throw new Error('ffmpeg failed to write the MP4')
    }
    if (errors.length) { console.error('page errors:\n  ' + errors.join('\n  ')); process.exitCode = 1 }
  } finally {
    await browser.close()
  }
}

main().catch((e) => { console.error('render: ' + e.message); process.exit(1) })
