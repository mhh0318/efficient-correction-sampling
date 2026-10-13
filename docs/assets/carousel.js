'use strict';
(() => {
  const root = document.querySelector('[data-sample-carousel]');
  if (!root) return;
  const assetBase = new URL('gallery/', document.currentScript.src);
  const viewport = root.querySelector('[data-carousel-window]');
  const track = root.querySelector('[data-carousel-track]');
  const toggle = root.querySelector('[data-carousel-toggle]');
  const status = root.querySelector('[data-carousel-status]');
  const tabs = [...root.querySelectorAll('[data-carousel-category]')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let collections, category, paused = reducedMotion.matches;

  function setPaused(value) {
    paused = value;
    root.dataset.carouselPaused = String(paused);
    toggle.textContent = paused ? 'Play' : 'Pause';
    toggle.setAttribute('aria-pressed', String(paused));
    toggle.setAttribute('aria-label', paused ? 'Start automatic sample scrolling' : 'Pause automatic sample scrolling');
    if (!paused) viewport.scrollLeft = 0;
  }
  setPaused(paused);
  toggle.addEventListener('click', () => setPaused(!paused));
  reducedMotion.addEventListener('change', event => setPaused(event.matches));
  viewport.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse') setPaused(true);
  });
  viewport.addEventListener('focusin', () => setPaused(true));

  function selectCategory(value) {
    category = collections[value] ? value : Object.keys(collections)[0];
    const collection = collections[category];
    // Consecutive previews, never ranked or filtered by image quality.
    const samples = collection.samples.slice(0, 64);
    const group = document.createElement('div');
    group.className = 'carousel-group';
    for (const sample of samples) {
      const link = document.createElement('a');
      link.className = 'carousel-sample';
      link.href = `gallery/?category=${encodeURIComponent(category)}&view=correction&sample=${encodeURIComponent(sample.id)}`;
      link.dataset.sampleId = sample.id;
      link.setAttribute('aria-label', `Open sample: ${sample.label || sample.prompt || collection.label}`);
      const image = document.createElement('img');
      image.src = new URL(sample.thumb, assetBase).href;
      image.alt = sample.label || sample.prompt || collection.label;
      image.width = image.height = 192;
      image.loading = 'lazy';
      image.decoding = 'async';
      link.append(image);
      group.append(link);
    }
    // The duplicate makes the loop seamless; it is omitted from keyboard navigation.
    const duplicate = group.cloneNode(true);
    duplicate.setAttribute('aria-hidden', 'true');
    duplicate.querySelectorAll('a').forEach(link => link.tabIndex = -1);
    track.replaceChildren(group, duplicate);
    track.style.setProperty('--carousel-duration', `${samples.length * 6}s`);
    track.getAnimations().forEach(animation => animation.currentTime = 0);
    viewport.scrollLeft = 0;
    tabs.forEach(tab => {
      const active = tab.dataset.carouselCategory === category;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    viewport.setAttribute('aria-label', collection.label);
    root.querySelector('[data-carousel-settings]').textContent = collection.settings || '';
    const browse = root.querySelector('[data-carousel-browse]');
    browse.href = `gallery/?category=${encodeURIComponent(category)}&view=correction`;
    browse.textContent = `Browse all ${collection.samples.length.toLocaleString()} samples ↗`;
    status.textContent = '';
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => { if (collections) selectCategory(tab.dataset.carouselCategory); });
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      const next = (index + (event.key === 'ArrowLeft' ? -1 : 1) + tabs.length) % tabs.length;
      tabs[next].focus();
      tabs[next].click();
    });
  });
  fetch(new URL('data.json', assetBase)).then(response => {
    if (!response.ok) throw new Error(`Gallery HTTP ${response.status}`);
    return response.json();
  }).then(data => {
    collections = data.collections;
    selectCategory(new URL(window.location.href).searchParams.get('category') || 'imagenet-conditional');
  }).catch(error => {
    console.error(error);
    status.textContent = 'The sample preview could not load. Please open the full gallery.';
  });
})();
