'use strict';
(() => {
  const dialog = document.getElementById('figure-viewer');
  let opener;
  document.querySelectorAll('[data-lightbox]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      opener = link;
      dialog.querySelector('img').src = link.href;
      dialog.querySelector('img').alt = link.querySelector('img').alt;
      dialog.showModal();
    });
  });
  dialog.querySelector('button').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });
  window.projectReady = true;
})();

// Count the published page; exclude local and tunnel previews.
(() => {
  if (window.location.hostname !== 'sunovivid.github.io') return;
  const script = document.createElement('script');
  script.type = 'module';
  script.src = 'https://static.cloudflareinsights.com/beacon.min.js';
  script.setAttribute('data-cf-beacon', JSON.stringify({token: 'bcb5d605a56943f49b556b03849d7327'}));
  document.body.appendChild(script);
})();
