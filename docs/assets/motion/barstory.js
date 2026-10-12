// Animated grouped bar chart that tells a result in phases (e.g. baseline -> + correction -> CFG).
// Markup (in page.md):
//   <figure class="barstory" data-ylabel="FID ↓">   (optional data-ymax, data-ystep)
//     <div class="barstory-series" data-name="Baseline (no CFG)" data-style="base" data-phase="Baseline, no guidance"></div>
//     <div class="barstory-series" data-name="Correction (no CFG)" data-style="corr" data-from="0" data-delta="percent"
//       data-phase="+ Correction, still no guidance"></div>
//     <div class="barstory-group" data-name="DiT-XL/2" data-values="9.58,2.07,2.51"></div>   (leave a value empty to
//       skip that series' bar in this group, e.g. "28.30,50.09,,")
//     <img class="barstory-fallback" ...>   (shown when JavaScript is off)
//   </figure>
// Each series is one phase; within a phase the groups animate in markup order. A series with data-from=i starts at
// series i's value and moves to its own, leaving a dashed outline of the change and a delta label
// (data-delta="percent" or "points"). Plays once when on screen; Replay restarts. Reduced motion: final state.
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1100, H = 440, L = 64, R = 18, T = 92, B = 52, GAP = 12;
  const GROW = 650, MOVE = 1100, STAGGER = 260, PAUSE = 650, FADE = 380;
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const clamp = (u) => Math.min(1, Math.max(0, u));

  for (const root of document.querySelectorAll('.barstory')) {
    const series = [...root.querySelectorAll('.barstory-series')].map((e) => ({
      name: e.dataset.name, style: e.dataset.style || 'base', phase: e.dataset.phase || e.dataset.name,
      from: e.dataset.from === undefined ? null : +e.dataset.from, delta: e.dataset.delta || 'percent',
    }));
    const groups = [...root.querySelectorAll('.barstory-group')].map((e) => ({
      name: e.dataset.name, values: e.dataset.values.split(',').map((v) => (v.trim() === '' ? NaN : +v)),
    }));
    if (!series.length || !groups.length) continue;
    root.classList.add('is-live');
    const all = groups.flatMap((g) => g.values).filter((v) => !Number.isNaN(v));
    const step = +root.dataset.ystep || (Math.max(...all) > 40 ? 20 : 4);
    const ymax = +root.dataset.ymax || Math.ceil(Math.max(...all) / step) * step;
    const PW = W - L - R, PH = H - T - B;
    const y = (v) => T + PH * (1 - v / ymax);
    const el = (tag, attrs, parent) => {
      const e = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
      if (parent) parent.append(e);
      return e;
    };

    const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', class: 'barstory-svg',
      'aria-label': groups.map((g) => `${g.name}: ` + series.map((s, i) => `${s.name} ${g.values[i]}`)
        .filter((_, i) => !Number.isNaN(g.values[i])).join(', ')).join('; ')});
    for (let v = 0; v <= ymax + 1e-9; v += step) {
      el('line', {x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'bs-grid'}, svg);
      el('text', {x: L - 10, y: y(v) + 5, 'text-anchor': 'end', class: 'bs-tick'}, svg).textContent = v;
    }
    el('text', {x: 18, y: T + PH / 2, 'text-anchor': 'middle', class: 'bs-axis', transform: `rotate(-90 18 ${T + PH / 2})`}, svg)
      .textContent = root.dataset.ylabel || '';
    const phaseText = el('text', {x: W / 2, y: 34, 'text-anchor': 'middle', class: 'bs-phase'}, svg);
    let lx = L;
    for (const s of series) {
      el('rect', {x: lx, y: 56, width: 14, height: 14, rx: 3, class: `bs-${s.style}`}, svg);
      el('text', {x: lx + 21, y: 68, class: 'bs-legend'}, svg).textContent = s.name;
      lx += 26 + s.name.length * 7.6 + 22;
    }

    const n = series.length;
    const gw = PW / groups.length;
    const BW = Math.min(54, (gw - 40 - GAP * (n - 1)) / n);
    const fmt = (v) => v.toFixed(2);
    const bars = groups.map((g, gi) => {
      const cx = L + gw * (gi + 0.5);
      const shown = series.map((_, si) => si).filter((si) => !Number.isNaN(g.values[si]));
      const x0 = cx - (shown.length * BW + (shown.length - 1) * GAP) / 2;
      const grp = el('g', {}, svg);
      el('text', {x: cx, y: y(0) + 30, 'text-anchor': 'middle', class: 'bs-group'}, grp).textContent = g.name;
      return series.map((s, si) => {
        if (Number.isNaN(g.values[si])) return null; // no bar for this series in this group
        const x = x0 + shown.indexOf(si) * (BW + GAP);
        const ghost = el('rect', {x, width: BW, rx: 4, class: 'bs-ghost', y: y(0), height: 0, opacity: 0});
        grp.prepend(ghost);
        return {
          ghost,
          rect: el('rect', {x, width: BW, rx: 4, class: `bs-${s.style}`, y: y(0), height: 0}, grp),
          text: el('text', {x: x + BW / 2, 'text-anchor': 'middle', class: `bs-val bs-${s.style}-t`, opacity: 0}, grp),
          delta: el('text', {x: x + BW / 2, 'text-anchor': 'middle', class: 'bs-delta', opacity: 0}, grp),
          // for an increase: where the bar started, drawn across the bar so the gain above it stands out
          mark: el('line', {x1: x + 3, x2: x + BW - 3, class: 'bs-mark', opacity: 0}, grp),
        };
      });
    });

    // timeline: one phase per series
    const starts = [];
    let t = 0;
    for (const s of series) {
      starts.push(t);
      t += (s.from === null ? GROW : FADE + MOVE) + STAGGER * (groups.length - 1) + PAUSE;
    }
    const END = t;

    function draw(ms) {
      const phase = starts.reduce((a, s0, i) => (ms >= s0 ? i : a), 0);
      phaseText.textContent = series[phase].phase;
      groups.forEach((g, gi) => series.forEach((s, si) => {
        const b = bars[gi][si];
        if (!b) return;
        const local = ms - starts[si] - gi * STAGGER;
        const dur = s.from === null ? GROW : MOVE;
        const u = ease(clamp((s.from === null ? local : local - FADE) / dur));
        const target = g.values[si];
        const origin = s.from === null ? 0 : g.values[s.from];
        const v = local < 0 ? 0 : origin + (target - origin) * u;
        const top = y(v);
        b.rect.setAttribute('y', top);
        b.rect.setAttribute('height', Math.max(0, y(0) - top));
        b.text.setAttribute('y', top - 8);
        b.text.textContent = fmt(v);
        // a moving bar fades in at its starting height instead of popping into place
        const appear = s.from === null ? 1 : clamp(local / FADE);
        b.rect.setAttribute('opacity', local < 0 ? 0 : appear);
        b.text.setAttribute('opacity', local < 0 ? 0 : s.from === null ? clamp(u * 3 - 2) : appear);
        if (s.from !== null) {
          // a decrease leaves a dashed outline of the removed part; an increase marks the starting level
          const change = target - origin;
          const a = y(Math.max(origin, v)), z = y(Math.min(origin, v));
          b.ghost.setAttribute('y', a);
          b.ghost.setAttribute('height', local < 0 || change >= 0 ? 0 : Math.max(0, z - a));
          b.ghost.setAttribute('opacity', local < 0 || change >= 0 ? 0 : 1);
          b.mark.setAttribute('y1', y(origin));
          b.mark.setAttribute('y2', y(origin));
          b.mark.setAttribute('opacity', local >= 0 && change > 0 && v - origin > 0.4 ? 1 : 0);
          b.delta.textContent = s.delta === 'points'
            ? `${change >= 0 ? '+' : '−'}${Math.abs(change).toFixed(2)}`
            : `${change >= 0 ? '+' : '−'}${Math.round(Math.abs(change / origin) * 100)}%`;
          b.delta.setAttribute('y', y(Math.max(origin, target)) - 30);
          b.delta.setAttribute('opacity', clamp((local - FADE - dur * 0.75) / 250));
        }
      }));
    }

    let start = 0, raf = 0, played = false;
    const frame = (now) => {
      const ms = now - start;
      draw(ms);
      if (ms < END) raf = requestAnimationFrame(frame);
    };
    const play = () => {
      cancelAnimationFrame(raf);
      played = true;
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return draw(END);
      start = performance.now();
      raf = requestAnimationFrame(frame);
    };
    const replay = document.createElement('button');
    replay.type = 'button';
    replay.className = 'barstory-replay';
    replay.textContent = 'Replay';
    replay.addEventListener('click', play);
    root.prepend(svg);
    svg.after(replay);
    draw(0);
    // start once at least ~40% of the chart is on screen; the observer can miss its first callback while the page is
    // still laying out, so scroll/resize/load also check directly until it has played
    const check = () => {
      if (played) return;
      const r = root.getBoundingClientRect();
      const seen = Math.min(r.bottom, innerHeight) - Math.max(r.top, 0);
      if (r.height > 0 && seen >= Math.min(r.height, innerHeight) * 0.4) {
        play();
        io.disconnect();
        removeEventListener('scroll', check);
        removeEventListener('resize', check);
      }
    };
    const io = new IntersectionObserver(check, {threshold: [0, 0.25, 0.5, 0.75, 1]});
    io.observe(root);
    addEventListener('scroll', check, {passive: true});
    addEventListener('resize', check);
    addEventListener('load', check);
    requestAnimationFrame(check);
  }
})();
