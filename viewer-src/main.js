/*! three.js (bundled into this report), licence:

The MIT License

Copyright © 2010-2024 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// scan-studio report viewer: before (raw dense points) and after (final model), a split compare view,
// the camera path, and a two-point measuring tool. All data comes from the JSON block in the page; every
// string from it is inserted with textContent, never as HTML.
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import * as panels from './panels.js'

const $ = (id) => document.getElementById(id)
const data = JSON.parse($('scan-data').textContent)
const report = data.report
const units = report.scale && report.scale.units === 'metres' ? 'm' : 'units'
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

function b64bytes (s) {
  const bin = atob(s)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}
function el (tag, cls, text) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = String(text)
  return e
}
function fmt (v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–'
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return v.toExponential(2)
  return Number(v).toFixed(digits)
}
function length (v) {
  if (units !== 'm') return `${fmt(v, 3)} units`
  return v < 1 ? `${fmt(v * 1000, 1)} mm` : `${fmt(v, 3)} m`
}

// ------------------------------------------------------------------ scene
const stage = $('stage')
const canvas = $('view')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.toneMapping = THREE.NeutralToneMapping
const css = getComputedStyle(document.documentElement)
const bg = new THREE.Color(css.getPropertyValue('--stage').trim() || '#e9e3d6')
const scene = new THREE.Scene()
scene.background = bg
const pmrem = new THREE.PMREMGenerator(renderer)
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1000)
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
controls.autoRotate = !reduceMotion
controls.autoRotateSpeed = 0.6

const size = report.model.size
const diag = Math.hypot(size[0], size[1], size[2]) || 1
const centre = new THREE.Vector3(0, size[1] / 2, 0)
camera.position.set(diag * 0.9, size[1] * 0.5 + diag * 0.55, diag * 1.35)
camera.near = diag / 200; camera.far = diag * 200; camera.updateProjectionMatrix()
controls.target.copy(centre)

const after = new THREE.Group()
const before = new THREE.Group()
scene.add(after, before)

// floor: a soft contact shadow disc and a fine grid, in the page's ink colour
const ink = new THREE.Color(css.getPropertyValue('--ink').trim() || '#1b1a17')
const grid = new THREE.GridHelper(diag * 3, 24, ink, ink)
grid.material.transparent = true; grid.material.opacity = 0.12
scene.add(grid)

// camera path, in the accent colour for cameras
const camColour = new THREE.Color(css.getPropertyValue('--cam').trim() || '#2944a8')
const camPts = report.sfm.cameras.map((c) => new THREE.Vector3(...c.centre))
const camPath = new THREE.Group()
if (camPts.length > 1) {
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(camPts),
    new THREE.LineBasicMaterial({ color: camColour, transparent: true, opacity: 0.7 }))
  camPath.add(line)
  const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints(camPts),
    new THREE.PointsMaterial({ color: camColour, size: diag * 0.018 }))
  camPath.add(dots)
}
camPath.visible = false
scene.add(camPath)

// before: the raw dense surface as points (floaters, table and all)
{
  const p = new Float32Array(b64bytes(data.before.points).buffer)
  const c = b64bytes(data.before.colors)
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(p, 3))
  g.setAttribute('color', new THREE.BufferAttribute(c, 3, true))
  const m = new THREE.PointsMaterial({ size: diag * 0.006, vertexColors: true })
  before.add(new THREE.Points(g, m))
}

let modelMesh = null
const wire = []
new GLTFLoader().parse(b64bytes(data.glb).buffer, '', (gltf) => {
  after.add(gltf.scene)
  const meshes = []
  gltf.scene.traverse((o) => { if (o.isMesh) meshes.push(o) })
  for (const o of meshes) {
    modelMesh = modelMesh || o
    const w = new THREE.Mesh(o.geometry, new THREE.MeshBasicMaterial({ color: ink, wireframe: true, transparent: true, opacity: 0.25 }))
    w.visible = false
    o.add(w); wire.push(w)
  }
  $('loading').hidden = true
}, (err) => { $('loading').textContent = 'The model could not be shown: ' + (err && err.message ? err.message : 'unknown error') })

// ------------------------------------------------------------------ modes
let mode = 'after'
let split = 0.5
const modeButtons = document.querySelectorAll('[data-mode]')
function setMode (m) {
  mode = m
  modeButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === m)))
  stage.dataset.mode = m
  $('divider').hidden = m !== 'compare'
}
modeButtons.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)))
function toggle (id, fn) {
  const b = $(id)
  b.addEventListener('click', () => { const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on)); fn(on) })
}
toggle('t-wire', (on) => wire.forEach((w) => { w.visible = on }))
toggle('t-cams', (on) => { camPath.visible = on })
toggle('t-spin', (on) => { controls.autoRotate = on })
if (reduceMotion) $('t-spin').setAttribute('aria-pressed', 'false')

// split divider for compare mode
const divider = $('divider')
let dragging = false
divider.addEventListener('pointerdown', (e) => { dragging = true; divider.setPointerCapture(e.pointerId) })
divider.addEventListener('pointerup', () => { dragging = false })
divider.addEventListener('pointercancel', () => { dragging = false })
divider.addEventListener('lostpointercapture', () => { dragging = false })
divider.addEventListener('pointermove', (e) => {
  if (!dragging) return
  const r = stage.getBoundingClientRect()
  split = Math.min(0.9, Math.max(0.1, (e.clientX - r.left) / r.width))
  divider.style.left = `${split * 100}%`
  divider.setAttribute('aria-valuenow', String(Math.round(split * 100)))
})
divider.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    split = Math.min(0.9, Math.max(0.1, split + (e.key === 'ArrowLeft' ? -0.05 : 0.05)))
    divider.style.left = `${split * 100}%`
  divider.setAttribute('aria-valuenow', String(Math.round(split * 100)))
    e.preventDefault()
  }
})

// ------------------------------------------------------------------ measure
const ray = new THREE.Raycaster()
const picks = []
const pickGroup = new THREE.Group()
scene.add(pickGroup)
const vermilion = new THREE.Color(css.getPropertyValue('--measure').trim() || '#c8401e')
let measuring = false
toggle('t-measure', (on) => { measuring = on; stage.classList.toggle('measuring', on); if (on) { picks.length = 0; pickGroup.clear(); showMeasure() } })
let down = null
canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY] })
canvas.addEventListener('pointerup', (e) => {
  if (!measuring || !modelMesh || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return
  const r = canvas.getBoundingClientRect()
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera)
  const hit = ray.intersectObject(modelMesh, false)[0]
  if (!hit) return
  if (picks.length === 2) { picks.length = 0; pickGroup.clear() }
  picks.push(hit.point.clone())
  const dot = new THREE.Mesh(new THREE.SphereGeometry(diag * 0.008, 16, 12), new THREE.MeshBasicMaterial({ color: vermilion, depthTest: false }))
  dot.renderOrder = 10
  dot.position.copy(hit.point); pickGroup.add(dot)
  if (picks.length === 2) {
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(picks), new THREE.LineBasicMaterial({ color: vermilion, depthTest: false }))
    l.renderOrder = 10
    pickGroup.add(l)
  }
  showMeasure()
})
function showMeasure () {
  const out = $('measure-out')
  out.replaceChildren()
  if (picks.length < 2) {
    out.append(el('p', 'hint', picks.length ? 'Now click the second point.' : 'Click two points on the model.'))
    return
  }
  const d = picks[0].distanceTo(picks[1])
  out.append(el('p', 'measure-value', length(d)))
  const p = (v) => `${fmt(v.x, 4)},${fmt(v.y, 4)},${fmt(v.z, 4)}`
  const cmd = `scan-studio scale ${report.out_name || report.name + '-out'} --points "${p(picks[0])}" "${p(picks[1])}" --length <metres>`
  out.append(el('p', 'hint', 'To set the real size from this pair, measure the same two points on the object and run:'))
  out.append(el('code', 'cmd', cmd))
}

// ------------------------------------------------------------------ render loop
function resize () {
  const r = stage.getBoundingClientRect()
  renderer.setSize(r.width, r.height, false)
  camera.aspect = r.width / Math.max(1, r.height)
  camera.updateProjectionMatrix()
}
new ResizeObserver(resize).observe(stage)
resize()
function draw () {
  const w = canvas.width / renderer.getPixelRatio()
  const h = canvas.height / renderer.getPixelRatio()
  if (mode === 'compare') {
    renderer.setScissorTest(true)
    before.visible = true; after.visible = false
    renderer.setScissor(0, 0, w * split, h); renderer.setViewport(0, 0, w, h); renderer.render(scene, camera)
    before.visible = false; after.visible = true
    renderer.setScissor(w * split, 0, w - w * split, h); renderer.render(scene, camera)
    renderer.setScissorTest(false)
  } else {
    before.visible = mode === 'before'; after.visible = mode === 'after'
    renderer.setViewport(0, 0, w, h); renderer.render(scene, camera)
  }
}
renderer.setAnimationLoop(() => { controls.update(); draw() })
const asked = new URLSearchParams(location.search).get('mode')
setMode(['after', 'before', 'compare'].includes(asked) ? asked : 'after')
panels.figures($('figures'), report)
panels.blurChart($('blur'), report)
panels.timings($('timings'), report)
panels.faces($('faces'), report)
panels.accuracy($('accuracy'), report)
window.__scanReady = () => Boolean(modelMesh)
// for the turntable capture: put the camera at a given angle round the model, the same height and distance
const home = camera.position.clone().sub(controls.target)
window.__scanAngle = (a) => {
  controls.autoRotate = false
  const r = Math.hypot(home.x, home.z)
  camera.position.set(controls.target.x + r * Math.sin(a), controls.target.y + home.y, controls.target.z + r * Math.cos(a))
  camera.lookAt(controls.target)
  controls.update()
}
window.__scanMode = setMode
