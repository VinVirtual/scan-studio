// The report's text panels: figures, the blur chart, stage timings, the face check and (when measured) accuracy.
// Everything is built with DOM calls and textContent; no string from the report is parsed as HTML.
const SVG = 'http://www.w3.org/2000/svg'

function el (tag, cls, text) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = String(text)
  return e
}
function svg (tag, attrs) {
  const e = document.createElementNS(SVG, tag)
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, String(v))
  return e
}
const n = (v, d = 0) => (v === null || v === undefined || Number.isNaN(Number(v)) ? '–' : Number(v).toLocaleString('en', { maximumFractionDigits: d, minimumFractionDigits: d }))

function figure (dl, label, value, note) {
  const wrap = el('div', 'fig')
  wrap.append(el('dt', '', label), el('dd', '', value))
  if (note) wrap.append(el('dd', 'note', note))
  dl.append(wrap)
}

export function figures (root, r) {
  const dl = el('dl', 'figures')
  const s = r.model.size
  const metric = r.scale && r.scale.units === 'metres'
  const dim = metric ? s.map((v) => n(v * 100, 1)).join(' × ') + ' cm' : s.map((v) => n(v, 3)).join(' × ') + ' units'
  figure(dl, 'Size (w × h × d)', dim, metric ? `scale from ${r.scale.method}` : 'no real size given')
  figure(dl, 'Triangles', n(r.model.triangles), `from ${n(r.dense.triangles)} raw`)
  figure(dl, 'Cameras placed', `${r.sfm.registered} / ${r.sfm.total}`, `${n(r.sfm.reprojection_error_px, 2)} px reprojection`)
  figure(dl, 'Frames kept', `${r.frames.chosen.length} / ${r.input.candidates}`, `${r.frames.blurry.length} blurry, ${r.frames.duplicates.length} still`)
  figure(dl, 'Sparse points', n(r.sfm.points), r.dense.method === 'sgbm-tsdf' ? `${r.dense.depth_maps} depth maps fused` : 'Poisson on sparse points')
  figure(dl, 'GLB', `${n(r.model.bytes / 1048576, 2)} MB`, r.model.texture ? `${r.model.texture}² texture` : 'vertex colours')
  figure(dl, 'Run time', `${n(r.timings.total_seconds / 60, 1)} min`, 'CPU only, no GPU')
  root.append(dl)
}

export function blurChart (root, r) {
  const scores = r.frames.scores
  if (!scores.length) return
  const W = 720
  const H = 150
  const pad = 4
  const max = Math.max(...scores) || 1
  const chosen = new Set(r.frames.chosen)
  const blurry = new Set(r.frames.blurry)
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Blur score of ${scores.length} candidate frames; ${chosen.size} chosen` })
  const bw = (W - pad * 2) / scores.length
  scores.forEach((v, i) => {
    const h = Math.max(1, (v / max) * (H - 26))
    const cls = chosen.has(i) ? 'bar chosen' : blurry.has(i) ? 'bar blurry' : 'bar'
    const rect = svg('rect', { x: pad + i * bw, y: H - 14 - h, width: Math.max(1, bw - (bw > 3 ? 1 : 0)), height: h, class: cls })
    const t = svg('title'); t.textContent = `frame ${i}: ${n(v, 1)}${chosen.has(i) ? ' (kept)' : blurry.has(i) ? ' (blurry)' : ''}`
    rect.append(t)
    s.append(rect)
  })
  if (r.frames.thresholds) {
    const pts = r.frames.thresholds.map((v, i) => `${pad + (i + 0.5) * bw},${H - 14 - (v / max) * (H - 26)}`).join(' ')
    s.append(svg('polyline', { points: pts, class: 'threshold' }))
  }
  s.append(svg('line', { x1: 0, x2: W, y1: H - 13.5, y2: H - 13.5, class: 'axis' }))
  const lab = svg('text', { x: pad, y: H - 2, class: 'axis-label' }); lab.textContent = 'start of capture'
  const lab2 = svg('text', { x: W - pad, y: H - 2, class: 'axis-label', 'text-anchor': 'end' }); lab2.textContent = 'end'
  s.append(lab, lab2)
  root.append(s)
  const key = el('p', 'key')
  key.append(el('span', 'k chosen', 'kept'), el('span', 'k plain', 'sharp, not needed'), el('span', 'k blurry', 'blurry'), el('span', 'k thr', 'blur threshold (share of local median)'))
  root.append(key)
}

export function timings (root, r) {
  const total = r.timings.total_seconds || 1
  const bar = el('div', 'tbar')
  const list = el('ol', 'tlist')
  r.timings.stages.forEach((st, i) => {
    const seg = el('span', `seg s${i % 6}`)
    seg.style.flexGrow = String(Math.max(st.seconds, total * 0.004))
    seg.title = `${st.stage}: ${n(st.seconds, 1)} s`
    bar.append(seg)
    const li = el('li')
    li.append(el('span', `sw s${i % 6}`), el('span', 'tname', st.stage), el('span', 'tsec', `${n(st.seconds, 1)} s`))
    list.append(li)
  })
  root.append(bar, list)
}

export function faces (root, r) {
  const f = r.faces
  const p = el('p', `verdict v-${f.verdict}`)
  p.append(el('strong', '', f.verdict === 'clear' ? 'No people found.' : f.verdict === 'warning' ? 'Check the frames.' : 'Confirmed object.'), document.createTextNode(' ' + f.message + '.'))
  root.append(p)
}

export function accuracy (root, r) {
  const a = r.accuracy
  if (!a) { root.closest('section').hidden = true; return }
  const dl = el('dl', 'figures')
  figure(dl, 'Mean distance', `${n(a.chamfer_mean_mm, 2)} mm`, `${n(a.chamfer_mean_pct, 2)} % of the object's size`)
  figure(dl, 'Median', `${n(a.median_mm, 2)} mm`, 'model to truth and truth to model')
  figure(dl, '95th percentile', `${n(a.p95_mm, 2)} mm`, 'the worst 5 % are further than this')
  figure(dl, `F-score at ${n(a.fscore_threshold_mm, 0)} mm`, n(a.fscore, 3), `precision ${n(a.precision, 3)}, recall ${n(a.recall, 3)}`)
  figure(dl, 'Height', `${n(a.height_model_mm, 1)} mm`, `truth ${n(a.height_truth_mm, 1)} mm`)
  figure(dl, 'Scale check', `${n(a.scale_ratio, 4)}`, 'size given ÷ size implied by the true camera path')
  root.append(dl)
  root.append(el('p', 'hint', a.method))
}
