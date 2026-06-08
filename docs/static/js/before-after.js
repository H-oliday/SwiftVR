/* ============================================================
   Before/After video slider + per-slider controls
   - clip-path masks the LQ video to reveal its left portion
   - slider aspect-ratio adapts to each video's true dimensions
     (so videos always show at their native size, no crop / no letterbox)
   - per-slider controls: play/pause, draggable progress, time readout
   - both videos seek together when user scrubs the progress bar
   ============================================================ */
(function () {

  /* -------- 1. video source resolution -------------------- */
  function applyVideoBase() {
    var base = window.SWIFTVR_VIDEO_BASE;
    if (typeof base !== 'string') base = 'static/videos/';
    if (base && !/[\/]$/.test(base)) base += '/';

    document.querySelectorAll('video[data-src]').forEach(function (v) {
      var name = v.getAttribute('data-src');
      if (!name) return;
      var newSrc = base + name;
      if (v.getAttribute('src') !== newSrc) {
        v.setAttribute('src', newSrc);
        try { v.load(); } catch (e) {}
        var p = v.play();
        if (p && typeof p.catch === 'function') p.catch(function () {});
      }
    });
  }

  /* -------- 2. utilities ---------------------------------- */
  function setClip(el, pct) {
    var inset = 'inset(0 ' + (100 - pct) + '% 0 0)';
    el.style.webkitClipPath = inset;
    el.style.clipPath = inset;
  }

  function formatTime(s) {
    if (!isFinite(s) || s < 0) s = 0;
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = Math.floor(s % 60);
    var pad = function (n) { return n < 10 ? '0' + n : '' + n; };
    if (h > 0) return h + ':' + pad(m) + ':' + pad(sec);
    return m + ':' + pad(sec);
  }

  /* -------- 3. aspect-ratio: match the video natively ------ */
  function syncAspectRatio(slider, video) {
    var apply = function () {
      if (video.videoWidth && video.videoHeight) {
        slider.style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
      }
    };
    if (video.readyState >= 1 && video.videoWidth) apply();
    video.addEventListener('loadedmetadata', apply);
  }

  /* -------- 4. main slider drag logic --------------------- */
  function initSlider(el) {
    var divider = el.querySelector('.va-divider');
    var handle  = el.querySelector('.va-handle');
    var before  = el.querySelector('.va-before');
    var after   = el.querySelector('.va-after');
    if (!divider || !handle || !before || !after) return;

    syncAspectRatio(el, after);

    el.tabIndex = 0;
    var dragging = false;

    function setPosition(pct) {
      pct = Math.max(0, Math.min(100, pct));
      setClip(before, pct);
      divider.style.left = pct + '%';
      handle.style.left  = pct + '%';
    }

    function pctFromEvent(e) {
      var rect = el.getBoundingClientRect();
      var x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      return (x / rect.width) * 100;
    }

    function start(e) {
      dragging = true;
      el.classList.add('is-dragging');
      setPosition(pctFromEvent(e));
      e.preventDefault();
    }
    function move(e) { if (dragging) setPosition(pctFromEvent(e)); }
    function end()   { dragging = false; el.classList.remove('is-dragging'); }

    el.addEventListener('mousedown', start);
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup',  end);
    el.addEventListener('touchstart', start, { passive: false });
    window.addEventListener('touchmove', move, { passive: false });
    window.addEventListener('touchend',  end);

    el.addEventListener('keydown', function (e) {
      var m = /(\d+(?:\.\d+)?)/.exec(before.style.clipPath || 'inset(0 50% 0 0)');
      var cur = m ? (100 - parseFloat(m[1])) : 50;
      if (e.key === 'ArrowLeft')  setPosition(cur - 3);
      if (e.key === 'ArrowRight') setPosition(cur + 3);
    });

    setPosition(50);

    // keep the two videos' time aligned during playback
    setInterval(function () {
      if (!before.paused && !after.paused) {
        if (Math.abs(before.currentTime - after.currentTime) > 0.15) {
          before.currentTime = after.currentTime;
        }
      }
    }, 800);

    var tryPlay = function () {
      var p1 = before.play(), p2 = after.play();
      if (p1 && p1.catch) p1.catch(function () {});
      if (p2 && p2.catch) p2.catch(function () {});
    };
    before.addEventListener('loadeddata', tryPlay);
    after.addEventListener('loadeddata', tryPlay);
  }

  /* -------- 5. per-slider controls ------------------------- */
  function initControls(slider) {
    var card = slider.closest('.va-card');
    if (!card) return;
    var controls = card.querySelector('[data-va-controls]');
    if (!controls) return;

    var before  = slider.querySelector('.va-before');
    var after   = slider.querySelector('.va-after');
    var playBtn = controls.querySelector('.va-btn-play');
    var progress= controls.querySelector('[data-va-progress]');
    var fill    = controls.querySelector('.va-progress-fill');
    var tCur    = controls.querySelector('.va-time-cur');
    var tTot    = controls.querySelector('.va-time-tot');
    var thumb   = controls.querySelector('.va-progress-thumb');
    if (!playBtn || !progress || !fill || !tCur || !tTot) return;

    /* --- play / pause --- */
    playBtn.addEventListener('click', function () {
      if (after.paused) {
        var p1 = after.play();  if (p1 && p1.catch) p1.catch(function () {});
        var p2 = before.play(); if (p2 && p2.catch) p2.catch(function () {});
      } else {
        after.pause(); before.pause();
      }
    });
    function updatePlayIcon() { playBtn.classList.toggle('is-playing', !after.paused); }
    after.addEventListener('play',  updatePlayIcon);
    after.addEventListener('pause', updatePlayIcon);
    updatePlayIcon();

    /* --- duration / current time --- */
    function updateDuration() {
      if (isFinite(after.duration)) tTot.textContent = formatTime(after.duration);
    }
    after.addEventListener('loadedmetadata', updateDuration);
    after.addEventListener('durationchange', updateDuration);
    updateDuration();

    function setProgressUI(pct) {
      fill.style.width = pct + '%';
      if (thumb) thumb.style.left = pct + '%';
    }

    function updateProgress() {
      if (!after.duration || !isFinite(after.duration)) return;
      var pct = (after.currentTime / after.duration) * 100;
      setProgressUI(pct);
      tCur.textContent = formatTime(after.currentTime);
    }
    after.addEventListener('timeupdate', updateProgress);

    /* --- seek by clicking / dragging (Pointer Events) ---
       Using setPointerCapture guarantees pointermove keeps firing
       on `progress` even after the cursor leaves the element,
       which is exactly what makes drag reliable. */
    var seeking = false;

    function seekFromEvent(e) {
      var rect = progress.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var pct = Math.max(0, Math.min(1, x / rect.width));
      if (!after.duration || !isFinite(after.duration)) return;
      var t = pct * after.duration;
      after.currentTime  = t;
      before.currentTime = t;
      setProgressUI(pct * 100);
      tCur.textContent = formatTime(t);
    }

    progress.addEventListener('pointerdown', function (e) {
      if (e.button !== undefined && e.button !== 0) return;     // left mouse only
      seeking = true;
      progress.classList.add('is-seeking');
      try { progress.setPointerCapture(e.pointerId); } catch (_) {}
      seekFromEvent(e);
      e.preventDefault();
    });

    progress.addEventListener('pointermove', function (e) {
      if (!seeking) return;
      seekFromEvent(e);
      e.preventDefault();
    });

    function endSeek(e) {
      if (!seeking) return;
      seeking = false;
      progress.classList.remove('is-seeking');
      if (e && e.pointerId !== undefined) {
        try { progress.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    }
    progress.addEventListener('pointerup',     endSeek);
    progress.addEventListener('pointercancel', endSeek);
    progress.addEventListener('lostpointercapture', endSeek);
  }

  /* -------- 6. bootstrap ---------------------------------- */
  function bootstrap() {
    applyVideoBase();
    document.querySelectorAll('[data-va-slider]').forEach(function (slider) {
      initSlider(slider);
      initControls(slider);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }
})();
