const tabs = document.querySelectorAll('[data-tab]');

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((candidate) => {
      const selected = candidate === tab;
      candidate.classList.toggle('active', selected);
      candidate.setAttribute('aria-selected', String(selected));
      const panel = document.getElementById(candidate.dataset.tab);
      panel.classList.toggle('active', selected);
      panel.hidden = !selected;
    });
  });
});

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copy);
    const value = source?.innerText || source?.textContent || '';
    if (!value.trim()) return;
    try {
      await navigator.clipboard.writeText(value.trim());
      const label = button.textContent;
      button.textContent = 'Copied';
      window.setTimeout(() => { button.textContent = label; }, 1400);
    } catch (_) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(source);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  });
});
