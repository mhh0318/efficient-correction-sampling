// Step-through controller for a rendered explainer video (e.g. the Method animation).
// Markup (in page.md): .mstep[data-src][data-poster] containing .mstep-beat[data-from][data-to] elements, each with a
// .mstep-title and a .mstep-line; times are seconds in the video. The controller adds the stage and controls.
// It autoplays the whole story while on screen (looping). The arrows (or the keyboard's left/right while the figure
// is on screen) play one beat and hold its last frame; the play button resumes the story; the bar scrubs, paused.
(() => {
  'use strict';
  const FRAME = 1 / 60;
  const LOOP_PAUSE = 1500; // ms on the last frame before the story starts over

  for (const root of document.querySelectorAll('.mstep')) {
    const beatEls = [...root.querySelectorAll('.mstep-beat')];
    const beats = beatEls.map((el) => ({el, from: +el.dataset.from, to: +el.dataset.to}));
    const total = beats[beats.length - 1].to;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // --- build stage and controls ---
    const words = root.querySelector('.mstep-words');
    words.setAttribute('aria-live', 'polite'); // (Pandoc rewrites aria-live in page.md to data-aria-live)
    const head = document.createElement('div');
    head.className = 'mstep-head';
    const prev = button('mstep-prev', 'Previous step', '<svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg>');
    const next = button('mstep-next', 'Next step', '<svg viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg>');
    words.replaceWith(head);
    head.append(prev, words, next);

    const stage = document.createElement('div');
    stage.className = 'mstep-stage';
    const video = document.createElement('video');
    Object.assign(video, {muted: true, playsInline: true, preload: 'auto', src: root.dataset.src, poster: root.dataset.poster || ''});
    video.setAttribute('muted', '');
    video.setAttribute('playsinline', '');
    video.setAttribute('aria-hidden', 'true');
    stage.append(video);

    const controls = document.createElement('div');
    controls.className = 'mstep-controls';
    const play = button('mstep-play', 'Play', '');
    const track = document.createElement('div');
    track.className = 'mstep-track';
    track.setAttribute('role', 'slider');
    track.setAttribute('aria-label', 'Step through the animation');
    track.setAttribute('aria-valuemin', '1');
    track.setAttribute('aria-valuemax', String(beats.length));
    track.tabIndex = 0;
    const fills = beats.map((b) => {
      const seg = document.createElement('div');
      seg.className = 'mstep-seg';
      seg.style.flexGrow = String(b.to - b.from);
      const fill = document.createElement('div');
      fill.className = 'mstep-fill';
      seg.append(fill);
      track.append(seg);
      return fill;
    });
    const count = document.createElement('span');
    count.className = 'mstep-count';
    controls.append(play, track, count);
    head.after(stage, controls);

    // --- state ---
    let mode = 'auto'; // 'auto': play the story; 'step': play one beat, then hold
    let target = null; // beat index being played in step mode
    let userPaused = reduceMotion;
    let visible = false;
    let loopTimer = 0;
    let shown = -1;

    const beatAt = (t) => {
      for (let i = beats.length - 1; i >= 0; i--) if (t >= beats[i].from - 1e-3) return i;
      return 0;
    };

    function render() {
      const t = video.currentTime || 0;
      const i = target ?? beatAt(t);
      if (i !== shown) {
        beatEls.forEach((el, k) => el.classList.toggle('is-active', k === i));
        count.textContent = `${i + 1} / ${beats.length}`;
        track.setAttribute('aria-valuenow', String(i + 1));
        track.setAttribute('aria-valuetext', beatEls[i].querySelector('.mstep-title').textContent);
        shown = i;
      }
      beats.forEach((b, k) => {
        const p = Math.min(1, Math.max(0, (t - b.from) / (b.to - b.from)));
        fills[k].style.transform = `scaleX(${p})`;
      });
      const playing = !video.paused && !video.ended;
      play.classList.toggle('is-playing', playing);
      play.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    }

    function loop() {
      if (mode === 'step' && target !== null && video.currentTime >= beats[target].to - FRAME) {
        video.pause();
        video.currentTime = Math.max(beats[target].from, beats[target].to - FRAME);
      }
      render();
      if (!video.paused) requestAnimationFrame(loop);
    }

    function start() {
      clearTimeout(loopTimer);
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
      requestAnimationFrame(loop);
    }

    function goTo(i) {
      clearTimeout(loopTimer);
      mode = 'step';
      target = (i + beats.length) % beats.length;
      video.currentTime = beats[target].from;
      if (reduceMotion) {
        video.currentTime = beats[target].to - FRAME;
        render();
      } else start();
    }

    function resume() {
      mode = 'auto';
      target = null;
      userPaused = false;
      if (video.currentTime >= total - 2 * FRAME) video.currentTime = 0;
      start();
    }

    prev.addEventListener('click', () => {
      const i = target ?? beatAt(video.currentTime);
      goTo(video.currentTime > beats[i].from + 0.8 && mode === 'auto' ? i : i - 1);
    });
    next.addEventListener('click', () => goTo((target ?? beatAt(video.currentTime)) + 1));
    play.addEventListener('click', () => {
      if (!video.paused) {
        video.pause();
        userPaused = true;
        render();
      } else resume();
    });
    stage.addEventListener('click', () => play.click());

    // scrub
    const seek = (e) => {
      const r = track.getBoundingClientRect();
      const u = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      video.pause();
      userPaused = true;
      mode = 'auto';
      target = null;
      video.currentTime = Math.min(total - FRAME, u * total);
      render();
    };
    track.addEventListener('pointerdown', (e) => {
      track.setPointerCapture(e.pointerId);
      seek(e);
      const move = (ev) => seek(ev);
      const up = () => {
        track.removeEventListener('pointermove', move);
        track.removeEventListener('pointerup', up);
      };
      track.addEventListener('pointermove', move);
      track.addEventListener('pointerup', up);
    });

    // keyboard: only while the figure is on screen and nothing else wants the keys
    document.addEventListener('keydown', (e) => {
      if (!visible || e.altKey || e.ctrlKey || e.metaKey) return;
      // with several step figures on screen, only the one nearest the viewport centre takes the keys
      const mid = innerHeight / 2;
      const dist = (el) => { const r = el.getBoundingClientRect(); return Math.abs((r.top + r.bottom) / 2 - mid); };
      const others = [...document.querySelectorAll('.mstep[data-visible]')].filter((el) => el !== root);
      if (others.some((el) => dist(el) < dist(root))) return;
      const tag = (document.activeElement && document.activeElement.tagName) || '';
      if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
      if (e.key === 'ArrowLeft') prev.click();
      else if (e.key === 'ArrowRight') next.click();
      else if (e.key === ' ' && root.contains(document.activeElement)) play.click();
      else return;
      e.preventDefault();
    });

    video.addEventListener('ended', () => {
      render();
      if (mode === 'auto' && !userPaused) {
        loopTimer = setTimeout(() => {
          if (visible && !userPaused && mode === 'auto') {
            video.currentTime = 0;
            start();
          }
        }, LOOP_PAUSE);
      }
    });
    video.addEventListener('seeked', render);
    video.addEventListener('loadedmetadata', () => {
      if (reduceMotion) video.currentTime = beats[0].to - FRAME;
      render();
    });

    // autoplay only while on screen
    new IntersectionObserver((entries) => entries.forEach((en) => {
      visible = en.isIntersecting;
      if (visible) root.dataset.visible = ''; else delete root.dataset.visible;
      if (visible && mode === 'auto' && !userPaused) start();
      else if (!visible && !video.paused) {
        video.pause();
        render();
      }
    }), {threshold: 0.35}).observe(root);

    render();
  }

  function button(cls, label, svg) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = cls;
    b.setAttribute('aria-label', label);
    b.innerHTML = svg;
    return b;
  }
})();
