/* ═══════════════════════════════════════════════════════════
   VONDETTA AI SYSTEMS — script.js
   East Africa's Premier Motor Insurance Intelligence Platform
═══════════════════════════════════════════════════════════ */

/* ── SMOOTH SCROLL ── */
function scrollTo(selector) {
  var el = document.querySelector(selector);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── HERO CANVAS PARTICLES ── */
(function () {
  var canvas = document.getElementById('heroCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W, H, particles = [];

  function resize() {
    W = canvas.width = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function Particle() {
    this.init(true);
  }
  Particle.prototype.init = function (scatter) {
    this.x = Math.random() * (W || 1200);
    this.y = scatter ? Math.random() * (H || 700) : (H || 700) + 10;
    this.vx = (Math.random() - 0.5) * 0.35;
    this.vy = -(Math.random() * 0.55 + 0.18);
    this.alpha = 0;
    this.maxAlpha = Math.random() * 0.55 + 0.15;
    this.r = Math.random() * 1.4 + 0.4;
    var palette = [
      '0,200,255',    // cyan
      '139,92,246',   // purple
      '59,130,246',   // blue
      '0,229,160',    // green
    ];
    this.color = palette[Math.floor(Math.random() * palette.length)];
  };
  Particle.prototype.update = function () {
    this.x += this.vx;
    this.y += this.vy;
    this.alpha = Math.min(this.alpha + 0.015, this.maxAlpha);
    if (this.y < -20) this.init(false);
  };
  Particle.prototype.draw = function () {
    ctx.save();
    ctx.globalAlpha = this.alpha;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + this.color + ',1)';
    ctx.fill();
    ctx.restore();
  };

  function drawConnections() {
    for (var i = 0; i < particles.length; i++) {
      for (var j = i + 1; j < particles.length; j++) {
        var dx = particles[i].x - particles[j].x;
        var dy = particles[i].y - particles[j].y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 110) {
          var a = (1 - dist / 110) * 0.07;
          ctx.save();
          ctx.globalAlpha = a;
          ctx.strokeStyle = '#00c8ff';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
          ctx.restore();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    drawConnections();
    for (var i = 0; i < particles.length; i++) {
      particles[i].update();
      particles[i].draw();
    }
    requestAnimationFrame(loop);
  }

  resize();
  for (var i = 0; i < 90; i++) particles.push(new Particle());
  loop();

  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  }, { passive: true });
})();

/* ── NAV SCROLL GLASS ── */
(function () {
  var nav = document.getElementById('mainNav');
  if (!nav) return;
  function onScroll() {
    if (window.scrollY > 60) nav.classList.add('scrolled');
    else nav.classList.remove('scrolled');
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

/* ── HAMBURGER MENU ── */
function toggleNav() {
  var links = document.getElementById('navLinks');
  var btn = document.getElementById('hamburger');
  if (!links || !btn) return;
  var open = links.classList.toggle('open');
  btn.classList.toggle('open', open);
}

/* ── STATS COUNTER (IntersectionObserver) ── */
(function () {
  var counters = document.querySelectorAll('.stat-num[data-target]');
  if (!counters.length) return;

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  function animate(el) {
    var target = parseFloat(el.dataset.target);
    var duration = parseInt(el.dataset.duration) || 2000;
    var decimal = parseInt(el.dataset.decimal) || 0;
    var prefix = el.dataset.prefix || '';
    var suffix = el.dataset.suffix || '';
    var start = performance.now();

    function step(now) {
      var elapsed = now - start;
      var progress = Math.min(elapsed / duration, 1);
      var val = easeOut(progress) * target;
      var display = decimal > 0 ? val.toFixed(decimal) : Math.floor(val).toLocaleString();
      el.textContent = prefix + display + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { animate(e.target); obs.unobserve(e.target); }
    });
  }, { threshold: 0.5 });

  counters.forEach(function (c) { obs.observe(c); });
})();

/* ── PROGRESS BARS ── */
(function () {
  var bars = document.querySelectorAll('.progress-fill[data-width]');
  if (!bars.length) return;

  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.style.width = e.target.dataset.width + '%';
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });

  bars.forEach(function (b) { obs.observe(b); });
})();

/* ── FADE-UP CARDS ON SCROLL ── */
(function () {
  var els = document.querySelectorAll(
    '.model-card,.product-card,.country-card,.step-card,.seg-card,.channel-card,.dm-kpi,.dm-chart-card'
  );
  if (!('IntersectionObserver' in window)) {
    els.forEach(function(el){ el.style.opacity='1'; el.style.transform='none'; });
    return;
  }

  els.forEach(function(el){
    el.style.opacity = '0';
    el.style.transform = 'translateY(22px)';
    el.style.transition = 'opacity .5s ease, transform .5s ease';
  });

  var staggerMap = {};
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      var parent = e.target.parentElement;
      var key = parent ? parent.className : 'root';
      staggerMap[key] = (staggerMap[key] || 0) + 1;
      var delay = Math.min((staggerMap[key] - 1) * 75, 350);
      setTimeout(function () {
        e.target.style.opacity = '1';
        e.target.style.transform = 'none';
      }, delay);
      obs.unobserve(e.target);
    });
  }, { threshold: 0.08 });

  els.forEach(function (el) { obs.observe(el); });
})();

/* ── MULTI-STEP QUOTE FORM ── */
var _step = 1;

function qNext(step) {
  document.getElementById('qStep' + _step).classList.remove('active');
  document.getElementById('fsb' + _step).classList.remove('active');
  document.getElementById('fsb' + _step).classList.add('done');
  _step = step;
  document.getElementById('qStep' + _step).classList.add('active');
  document.getElementById('fsb' + _step).classList.add('active');
}

function qPrev(step) {
  document.getElementById('qStep' + _step).classList.remove('active');
  document.getElementById('fsb' + _step).classList.remove('active');
  var prev = _step;
  _step = step;
  document.getElementById('qStep' + _step).classList.add('active');
  document.getElementById('fsb' + _step).classList.add('active');
  document.getElementById('fsb' + prev).classList.remove('done');
}

/* ── QUOTE CALCULATOR ── */
function runQuote() {
  var btn = document.getElementById('calcBtn');
  btn.textContent = 'AI analysing…';
  btn.disabled = true;

  setTimeout(function () {
    var vehicleVal  = parseFloat(document.getElementById('q_vehicleValue').value)  || 1_500_000;
    var premium     = parseFloat(document.getElementById('q_premium').value)        || 45_000;
    var driverAge   = parseInt(document.getElementById('q_driverAge').value)        || 34;
    var tenure      = parseFloat(document.getElementById('q_tenure').value)         || 2;
    var pastClaims  = parseInt(document.getElementById('q_pastClaims').value)       || 0;
    var recentClaims= parseInt(document.getElementById('q_recentClaims').value)     || 0;
    var claimAmt    = parseFloat(document.getElementById('q_claimAmt').value)       || 0;
    var coverage    = document.getElementById('q_coverage').value;
    var channel     = (document.querySelector('input[name="q_channel"]:checked') || {}).value || 'agent';

    /* ── simplified model heuristics mirroring XGBoost patterns ── */
    var churn = 0.14;
    if (tenure < 2)             churn += 0.20;  // early tenure spike
    if (tenure >= 1 && tenure <= 3) churn += 0.08; // valley of death
    if (pastClaims > 2)         churn += 0.12;
    if (driverAge < 26)         churn += 0.10;
    if (channel === 'broker')   churn += 0.07;
    churn = Math.min(churn, 0.94);

    var claimsProb = 0.18;
    if (pastClaims > 0)   claimsProb += pastClaims * 0.055;
    if (recentClaims > 0) claimsProb += 0.14;
    if (driverAge < 26)   claimsProb += 0.09;
    claimsProb = Math.min(claimsProb, 0.92);

    /* CLV tier */
    var estCLV = premium * Math.max(tenure, 1) * (1 - churn) * 1.35;
    var tier = 'Bronze';
    if (estCLV > 4500) tier = 'Platinum';
    else if (estCLV > 3600) tier = 'Gold';
    else if (estCLV > 2700) tier = 'Silver';

    /* base premium from vehicle value */
    var rates = { comprehensive: 0.034, fire_theft: 0.022, third_party: 0.009 };
    var rate = rates[coverage] || 0.034;
    var quoted = vehicleVal * rate;
    if (claimsProb > 0.40) quoted *= 1.14;       // high-risk loading
    if (churn > 0.50)      quoted *= 0.94;        // retention discount
    if (tier === 'Platinum' || tier === 'Gold') quoted *= 0.92;

    /* loss ratio */
    var lossRatio = claimAmt > 0 ? ((claimAmt / premium) * 100).toFixed(0) : null;

    /* recommendations */
    var recs = [];
    if (tenure < 2)       recs.push('Early-tenure loyalty bonus applied');
    if (churn > 0.40)     recs.push('Proactive retention agent auto-assigned');
    if (claimsProb < 0.22) recs.push('Clean record discount: low-risk driver');
    if (tier === 'Platinum') recs.push('Platinum CLV tier — concierge service');
    if (tier === 'Gold')  recs.push('Gold tier — priority claims handling');
    if (!recs.length)     recs.push('Standard AI-optimised pricing applied');

    /* render result */
    var churnColor  = churn < 0.30  ? 'var(--green)' : churn < 0.60 ? 'var(--gold)' : 'var(--red)';
    var claimsColor = claimsProb < 0.30 ? 'var(--green)' : claimsProb < 0.55 ? 'var(--gold)' : 'var(--red)';
    var coverLabel  = coverage.replace('_', ' ').replace(/\b\w/g, function(l){ return l.toUpperCase(); });

    var html = [
      '<div class="quote-result-inner">',
        '<div class="qr-score">',
          '<div class="qr-tier">' + tier + ' Tier · ' + coverLabel + ' Cover</div>',
          '<div class="qr-amt">KES ' + Math.round(quoted).toLocaleString() + '</div>',
          '<div class="qr-period">Estimated Annual Premium</div>',
        '</div>',
        '<div class="qr-metrics">',
          '<div class="qr-metric">',
            '<span class="qrm-val" style="color:' + churnColor + '">' + (churn * 100).toFixed(0) + '%</span>',
            '<span class="qrm-lbl">Churn Risk</span>',
          '</div>',
          '<div class="qr-metric">',
            '<span class="qrm-val" style="color:' + claimsColor + '">' + (claimsProb * 100).toFixed(0) + '%</span>',
            '<span class="qrm-lbl">Claims Probability</span>',
          '</div>',
          lossRatio ? (
            '<div class="qr-metric">' +
              '<span class="qrm-val" style="color:var(--cyan)">' + lossRatio + '%</span>' +
              '<span class="qrm-lbl">Loss Ratio</span>' +
            '</div>'
          ) : '',
          '<div class="qr-metric">',
            '<span class="qrm-val" style="color:var(--gold)">KES ' + Math.round(estCLV).toLocaleString() + '</span>',
            '<span class="qrm-lbl">Est. Customer CLV</span>',
          '</div>',
        '</div>',
        '<div class="qr-recs">',
          '<p>AI Recommendations</p>',
          '<ul>' + recs.map(function(r){ return '<li>' + r + '</li>'; }).join('') + '</ul>',
        '</div>',
      '</div>'
    ].join('');

    document.getElementById('quoteResult').innerHTML = html;
    btn.textContent = 'Calculate My Quote →';
    btn.disabled = false;
    qNext(3);

  }, 1700);
}

function applyNow() {
  alert('Full application flow connects to the Vondetta backend API at /api/v1/applications/new.\nThis demo displays the quote engine only.');
}

/* ── DASHBOARD SIDEBAR ── */
(function () {
  var items = document.querySelectorAll('.dms-item');
  items.forEach(function (item) {
    item.addEventListener('click', function () {
      items.forEach(function (i) { i.classList.remove('active'); });
      item.classList.add('active');
    });
  });
})();

/* ── ACTIVE NAV LINK ON SCROLL ── */
(function () {
  var sections = document.querySelectorAll('section[id]');
  var navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  if (!sections.length || !navLinks.length) return;

  function setActive() {
    var scrollY = window.scrollY + 120;
    sections.forEach(function (sec) {
      if (scrollY >= sec.offsetTop && scrollY < sec.offsetTop + sec.offsetHeight) {
        navLinks.forEach(function (a) {
          a.style.color = '';
          if (a.getAttribute('href') === '#' + sec.id) a.style.color = 'var(--cyan)';
        });
      }
    });
  }
  window.addEventListener('scroll', setActive, { passive: true });
  setActive();
})();

/* ── CLOSE MOBILE NAV ON LINK CLICK ── */
(function () {
  document.querySelectorAll('.nav-links a').forEach(function (a) {
    a.addEventListener('click', function () {
      var links = document.getElementById('navLinks');
      var btn = document.getElementById('hamburger');
      if (links) links.classList.remove('open');
      if (btn) btn.classList.remove('open');
    });
  });
})();
