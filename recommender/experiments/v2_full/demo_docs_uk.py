"""Ukrainian light-theme Presentation and Documentation HTML generators for the portable B2B demo packet.

Both pages are completely self-contained (zero external fonts/CDNs/internet required), strictly light-theme,
Ukrainian-only, and link seamlessly with ``OPEN_DEMO.html`` inside the ``B2B_Recommender_Demo/`` ZIP folder.
"""

from __future__ import annotations

from typing import Mapping


PRESENTATION_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Презентація — Система AI-рекомендацій B2B</title>
<style>
  :root {
    color-scheme: light;
    --bg: #eef2f7;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #2563eb;
    --primary-soft: rgba(37, 99, 235, 0.10);
    --green: #059669;
    --green-bg: rgba(16, 185, 129, 0.12);
    --green-border: #10b981;
    --amber: #b45309;
    --amber-bg: rgba(245, 158, 11, 0.14);
    --amber-border: #f59e0b;
    --purple: #6d28d9;
    --purple-bg: rgba(124, 58, 237, 0.10);
    --shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px 20px 28px;
    gap: 14px;
  }
  .nav-bar {
    width: 100%; max-width: 1180px;
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 10px 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
  }
  .nav-links { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .nav-pill {
    text-decoration: none; font-size: 13px; font-weight: 600; color: var(--muted);
    padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface-2);
    transition: all 0.15s;
  }
  .nav-pill:hover { border-color: var(--primary); color: var(--primary); }
  .nav-pill.active { background: var(--primary); color: #fff; border-color: var(--primary); }

  .deck-controls { display: flex; align-items: center; gap: 8px; }
  .btn {
    border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
    font-family: var(--sans); font-size: 13px; font-weight: 600;
    padding: 6px 12px; border-radius: 8px; cursor: pointer;
  }
  .btn:hover { border-color: var(--primary); color: var(--primary); }
  .btn.primary { background: var(--primary); color: #fff; border-color: var(--primary); }
  .btn.primary:hover { opacity: 0.92; color: #fff; }
  .counter { font-family: var(--mono); font-size: 12.5px; font-weight: 700; color: var(--muted); min-width: 64px; text-align: center; }

  .progress-wrap { width: 100%; max-width: 1180px; height: 5px; background: #dbe3ee; border-radius: 999px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--primary); transition: width 0.25s ease; }

  .stage { width: 100%; max-width: 1180px; flex: 1; display: flex; flex-direction: column; }
  .slide {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: var(--shadow);
    padding: 40px 48px;
    min-height: 620px;
    flex-direction: column;
    justify-content: space-between;
    gap: 24px;
  }
  .slide.active { display: flex; }

  .slide-kicker {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--primary); background: var(--primary-soft);
    padding: 5px 12px; border-radius: 999px; width: fit-content;
  }
  .slide h1 { font-size: 38px; line-height: 1.15; letter-spacing: -0.025em; margin-top: 14px; }
  .slide h2 { font-size: 28px; line-height: 1.2; letter-spacing: -0.02em; margin-top: 10px; }
  .lead { font-size: 17px; color: var(--muted); line-height: 1.6; max-width: 860px; margin-top: 10px; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .card {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px;
  }
  .card.accent { background: var(--primary-soft); border-color: rgba(37, 99, 235, 0.25); }
  .card.green { background: var(--green-bg); border-color: rgba(16, 185, 129, 0.35); }
  .card.amber { background: var(--amber-bg); border-color: rgba(245, 158, 11, 0.40); }
  .card h3 { font-size: 16px; font-weight: 800; margin-bottom: 6px; }
  .card p, .card li { font-size: 14px; color: #334155; line-height: 1.55; }
  .card ul { padding-left: 18px; display: flex; flex-direction: column; gap: 5px; }

  .kpi-big { font-size: 36px; font-weight: 800; font-family: var(--mono); letter-spacing: -0.02em; }
  .kpi-big.green { color: var(--green); }
  .kpi-big.blue { color: var(--primary); }
  .kpi-big.amber { color: var(--amber); }
  .kpi-lbl { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 6px; }
  .kpi-sub { font-size: 13px; color: var(--muted); margin-top: 4px; }

  .timeline { display: grid; grid-template-columns: 2.2fr 1fr 1fr; gap: 12px; }
  .t-seg { border-radius: 12px; padding: 16px 18px; border: 1px solid var(--border); }
  .t-train { background: #eff6ff; border-color: #93c5fd; }
  .t-val { background: #ecfdf5; border-color: #6ee7b7; }
  .t-test { background: #f8fafc; border-color: #cbd5e1; border-style: dashed; }

  .evo-list { display: flex; flex-direction: column; gap: 10px; }
  .evo-row { display: grid; grid-template-columns: 270px 1fr 82px 86px; gap: 12px; align-items: center; font-size: 13.5px; }
  .evo-track { height: 14px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
  .evo-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #60a5fa, #2563eb); }
  .evo-row.best .evo-fill { background: linear-gradient(90deg, #10b981, #059669); }
  .evo-row.best { font-weight: 800; }

  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); padding: 10px 12px; border-bottom: 2px solid var(--border); }
  td { padding: 12px 12px; border-bottom: 1px solid var(--border); }
  .mono { font-family: var(--mono); }

  .svg-box {
    background: #fcfdff; border: 1px solid var(--border); border-radius: 14px; padding: 14px;
    display: flex; justify-content: center; align-items: center;
  }
  .svg-box svg { width: 100%; height: auto; max-height: 360px; }

  .slide-foot {
    display: flex; justify-content: space-between; align-items: center;
    padding-top: 14px; border-top: 1px solid var(--border);
    font-size: 12px; color: var(--muted);
  }
  .dots { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; max-width: 1180px; }
  .dot {
    border: 1px solid var(--border); background: var(--surface); color: var(--muted);
    font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 999px; cursor: pointer;
  }
  .dot.active { background: var(--primary); color: #fff; border-color: var(--primary); }

  @media (max-width: 960px) {
    .slide { padding: 24px 22px; min-height: auto; }
    .slide h1 { font-size: 28px; }
    .grid-2, .grid-3, .grid-4, .timeline { grid-template-columns: 1fr; }
    .evo-row { grid-template-columns: 170px 1fr 64px 68px; }
  }
  @media print {
    body { background: #fff; padding: 0; }
    .nav-bar, .progress-wrap, .dots { display: none !important; }
    .slide { display: flex !important; box-shadow: none; border: 1px solid #cbd5e1; page-break-after: always; margin-bottom: 18px; min-height: 92vh; }
  }
</style>
</head>
<body>

<div class="nav-bar">
  <div class="nav-links">
    <a class="nav-pill" href="OPEN_DEMO.html">Інтерактивне демо</a>
    <a class="nav-pill active" href="PRESENTATION.html">Презентація</a>
    <a class="nav-pill" href="DOCUMENTATION.html">Документація та архітектура</a>
  </div>
  <div class="deck-controls">
    <button class="btn" id="btn-prev">← Назад</button>
    <span class="counter" id="slide-counter">1 / 9</span>
    <button class="btn primary" id="btn-next">Далі →</button>
    <button class="btn" onclick="window.print()">Друк / PDF</button>
  </div>
</div>

<div class="progress-wrap"><div class="progress-fill" id="progress-fill" style="width:11.1%"></div></div>

<div class="stage">
  <!-- SLIDE 1 -->
  <section class="slide active">
    <div>
      <div class="slide-kicker">Фінальний реліз · Модель w2_2mem_recency_hl365</div>
      <h1>Система AI-рекомендацій товарів для B2B-продажів</h1>
      <p class="lead">
        Персоналізована нейромережа, яка для кожного бізнес-клієнта за <strong>3 мілісекунди</strong>
        оцінює весь каталог із <strong>1 141 продукту</strong> та формує Топ-10 рекомендацій:
        і регулярні повторні замовлення, і нові товари, які клієнт ще ніколи не купував.
      </p>
    </div>
    <div class="grid-4">
      <div class="card green">
        <div class="kpi-lbl">Точність Топ-10 (Recall@10)</div>
        <div class="kpi-big green">82.26%</div>
        <div class="kpi-sub">На 16 623 реальних замовленнях</div>
      </div>
      <div class="card accent">
        <div class="kpi-lbl">Якість порядку (NDCG@10)</div>
        <div class="kpi-big blue">65.48%</div>
        <div class="kpi-sub">Точні товари — на перших рядках</div>
      </div>
      <div class="card amber">
        <div class="kpi-lbl">Чисті новинки (Cross-Sell)</div>
        <div class="kpi-big amber">52.27%</div>
        <div class="kpi-sub">2 350 із 4 496 нових покупок</div>
      </div>
      <div class="card">
        <div class="kpi-lbl">Швидкість роботи (CPU)</div>
        <div class="kpi-big">2.9 мс</div>
        <div class="kpi-sub">Без GPU · 1.13M параметрів</div>
      </div>
    </div>
    <div class="slide-foot">
      <span>Керування: стрілки ← / → на клавіатурі</span>
      <span>Слайд 1 із 9</span>
    </div>
  </section>

  <!-- SLIDE 2 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Бізнес-задача</div>
      <h2>Навіщо потрібна AI-система рекомендацій у B2B</h2>
      <p class="lead">
        У каталозі понад 1 660 позицій (SKU) різних фасувань, в'язкостей та брендів.
        Менеджеру фізично неможливо тримати в голові повну історію кожного клієнта, ритм його закупівель
        та попит схожих підприємств у регіоні.
      </p>
    </div>
    <div class="grid-3">
      <div class="card green">
        <h3>1. Регулярні повтори</h3>
        <p>
          Клієнт періодично докуповує звичні мастила й витратники. Модель відстежує індивідуальний
          цикл замовлень кожного товару і підказує саме тоді, коли запаси закінчуються.
        </p>
      </div>
      <div class="card amber">
        <h3>2. Нові товари (Крос-продажі)</h3>
        <p>
          Найскладніше — запропонувати продукт, який клієнт <strong>ніколи раніше не брав</strong>,
          але який купують схожі компанії з його галузі, регіону або парку техніки.
        </p>
      </div>
      <div class="card accent">
        <h3>3. Нові клієнти (1-ше замовлення)</h3>
        <p>
          Навіть якщо клієнт має <strong>0 минулих покупок</strong>, система спирається на опис компанії,
          галузь, регіон та актуальний попит менеджера й вгадує <strong>57.04%</strong> позицій у Топ-10.
        </p>
      </div>
    </div>
    <div class="slide-foot">
      <span>Один інструмент закриває всі три сценарії продажів</span>
      <span>Слайд 2 із 9</span>
    </div>
  </section>

  <!-- SLIDE 3 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Чесна валідація без підглядання</div>
      <h2>Як ми перевіряємо модель на реальних історичних даних</h2>
      <p class="lead">
        Усі дані розділені строго за часом. Щоб передбачити замовлення клієнта на певну дату,
        модель бачить <strong>лише ті покупки, ціни, розрахунки та рейтинги, які відбулися ДО цієї дати</strong>.
      </p>
    </div>
    <div class="timeline">
      <div class="t-seg t-train">
        <div class="kpi-lbl" style="color:#1d4ed8;">1. Навчання (Train)</div>
        <div class="mono" style="font-size:20px;font-weight:800;margin:6px 0;">&lt; 2026-02-01</div>
        <p style="font-size:14px;color:#334155;"><strong>137 765 замовлень</strong> — на цих даних нейромережа вивчала закономірності покупок із акцентом на свіжіші 2025–2026 продажі.</p>
      </div>
      <div class="t-seg t-val">
        <div class="kpi-lbl" style="color:#047857;">2. Валідація (Демо)</div>
        <div class="mono" style="font-size:20px;font-weight:800;margin:6px 0;">2026-02 → 2026-06</div>
        <p style="font-size:14px;color:#334155;"><strong>16 623 замовлення</strong> (4 496 нових покупок). Саме з цього періоду взяті всі кейси в інтерактивному демо.</p>
      </div>
      <div class="t-seg t-test" style="background:#fffbeb;border-color:#f59e0b;border-style:solid;">
        <div class="kpi-lbl" style="color:#b45309;">3. Фінальний тест (Демо)</div>
        <div class="mono" style="font-size:20px;font-weight:800;margin:6px 0;">2026-07 → 11-20</div>
        <p style="font-size:14px;color:#334155;"><strong>13 184 замовлення</strong> (5–10 місяців після навчання). Модель ніколи не бачила цих дат — саме звідси взяті кейси в <code>OPEN_DEMO.html</code>.</p>
      </div>
    </div>
    <div class="card">
      <strong>Чому це важливо:</strong> жодна оцінка не завищена. Навіть через 5–10 місяців після завершення навчання (2026-02-01) заморожена модель показала <strong>79.21% Recall@10</strong> (модельований каталог) / <strong>76.63%</strong> (повний каталог) на незалежному тесті.
    </div>
    <div class="slide-foot">
      <span>Суворий хронологічний контроль (Zero Future Leakage)</span>
      <span>Слайд 3 із 9</span>
    </div>
  </section>

  <!-- SLIDE 4 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Конвеєр рекомендацій</div>
      <h2>Як народжується прогноз Топ-10: 4 послідовні кроки</h2>
      <p class="lead">
        Замість того щоб порівнювати тисячі однакових каністр різного об'єму, система спочатку мислить
        на рівні формуляцій (продуктів), а наприкінці підбирає ідеальну тару для конкретного клієнта.
      </p>
    </div>
    <div class="grid-4">
      <div class="card">
        <div class="kpi-lbl" style="color:var(--primary);">Крок 1</div>
        <h3>Уніфікація каталогу</h3>
        <p>1 660 SKU об'єднано в <strong>1 141 канонічний продукт</strong>. У Топ-10 ніколи не з'являться 3 однакові оливи лише у різній тарі (4л, 20л, 209л).</p>
      </div>
      <div class="card">
        <div class="kpi-lbl" style="color:var(--primary);">Крок 2</div>
        <h3>42 сигнали на дату</h3>
        <p>Для кожного з 1 141 доступних товарів миттєво обчислюються 42 каузальні ознаки: цикли повторів, схожі клієнти, регіон, менеджер, галузь, в'язкість.</p>
      </div>
      <div class="card accent">
        <div class="kpi-lbl" style="color:var(--primary);">Крок 3</div>
        <h3>Нейромережа з 2 головами</h3>
        <p>Одна голова оцінює ймовірність <strong>повтору</strong>, інша спеціалізується на <strong>нових товарах</strong>. Калібрування поєднує їх у чесний Топ-10.</p>
      </div>
      <div class="card green">
        <div class="kpi-lbl" style="color:var(--green);">Крок 4</div>
        <h3>Підбір фасування (SKU)</h3>
        <p>Для кожного продукту з Топ-10 система автоматично обирає конкретний артикул (бочка 209л, відро 20л чи каністра 4л) за звичками клієнта.</p>
      </div>
    </div>
    <div class="slide-foot">
      <span>Повний цикл займає ~3 мс на звичайному процесорі (CPU)</span>
      <span>Слайд 4 із 9</span>
    </div>
  </section>

  <!-- SLIDE 5: INTERACTIVE 3D ARCHITECTURE -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Інтерактивна 3D-архітектура нейромережі</div>
      <h2>З чого складається модель Wave2BE2ERanker (3D-вигляд)</h2>
      <p class="lead">
        Одностадійна наскрізна (End-to-End) архітектура: потягніть мишкою, щоб покрутити 3D-структуру шарів, або натисніть на будь-який блок.
      </p>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;">
      <div class="svg-box" style="padding:8px;position:relative;">
        <canvas id="pres-arch3d" style="width:100%;height:310px;display:block;cursor:grab;border-radius:10px;"></canvas>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);">
        <span id="pres-3d-info"><strong>Підказка:</strong> перетягуйте для обертання 3D · клікніть по блоку, щоб побачити його роль.</span>
        <div style="display:flex;gap:6px;">
          <button class="btn" id="pres-3d-auto" style="padding:4px 10px;font-size:12px;">Авто-обертання: Увімк.</button>
          <button class="btn" id="pres-3d-reset" style="padding:4px 10px;font-size:12px;">Скинути ракурс</button>
        </div>
      </div>
    </div>
    <div class="slide-foot">
      <span>Повний інтерактивний 3D-розбір кожного блоку — у файлі DOCUMENTATION.html</span>
      <span>Слайд 5 із 9</span>
    </div>
  </section>

  <!-- SLIDE 6 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Ключ до високої точності</div>
      <h2>42 каузальні сигнали + акцент на свіжі продажі (365 днів)</h2>
      <p class="lead">
        Нейромережа не вгадує навмання — вона поєднує глибокий аналіз історії з 42 перевіреними
        бізнес-сигналами та надає більшої ваги свіжим тенденціям ринку 2025–2026 років.
      </p>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3>Що входить у 42 сигнали (6 груп)</h3>
        <ul>
          <li><strong>Історія та цикл повторів:</strong> скільки разів клієнт купував товар, скільки днів минуло та чи настав «час поповнення».</li>
          <li><strong>Схожі клієнти (CF):</strong> що купують інші компанії з подібним кошиком товарів.</li>
          <li><strong>Зв'язки товарів (Sequence Lift):</strong> які продукти найчастіше беруть слідом один за одним.</li>
          <li><strong>Менеджер і регіон (30д / 90д):</strong> що активно продається у цьому регіоні та галузі прямо зараз.</li>
          <li><strong>Галузь, техніка і в'язкість:</strong> сумісність специфікацій (SAE/ISO, сфера застосування, тип тари).</li>
          <li><strong>Фінанси клієнта (CUST_7):</strong> частка передплати, відстрочка, дисципліна оплат.</li>
        </ul>
      </div>
      <div class="card green">
        <h3>Чому працює вага свіжості (Half-Life = 365 днів)</h3>
        <p style="margin-bottom:12px;">
          Асортимент і попит змінюються з часом. Замість того щоб вважати замовлення 2022 року
          таким же важливим, як замовлення 2026 року, під час навчання застосовується плавне згасання ваги:
        </p>
        <ul>
          <li><strong>Свіжі продажі (останні місяці):</strong> вага <strong>1.00</strong> (максимальний вплив)</li>
          <li><strong>1 рік тому:</strong> вага <strong>0.50</strong></li>
          <li><strong>2 роки тому:</strong> вага <strong>0.25</strong></li>
          <li><strong>Давніші роки:</strong> мінімальна базова вага <strong>0.10</strong> (база знань зберігається)</li>
        </ul>
        <p style="margin-top:12px;"><strong>Результат:</strong> найвища точність відкриття новинок (<strong>+0.89 п.п.*</strong> статистично значущий приріст) без втрати загального Recall@10.</p>
      </div>
    </div>
    <div class="slide-foot">
      <span>Усі 42 сигнали розраховуються строго до дати замовлення</span>
      <span>Слайд 6 із 9</span>
    </div>
  </section>

  <!-- SLIDE 7 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Еволюція якості</div>
      <h2>Як покращувалася модель крок за кроком</h2>
      <p class="lead">
        Порівняння всіх ключових етапів розробки: загальне покриття замовлень у Топ-10 (<strong>Recall@10</strong>)
        та частка вгаданих <strong>нових</strong> для клієнта товарів (<strong>New@10</strong>).
      </p>
    </div>
    <div class="card">
      <div class="evo-list">
        <div class="evo-row"><div>1. Початковий SKU-бейзлайн (v1)</div><div class="evo-track"><div class="evo-fill" style="width:24%"></div></div><div class="mono" style="text-align:right;">76.15%</div><div class="mono" style="text-align:right;color:var(--amber);">11.25%</div></div>
        <div class="evo-row"><div>2. Уніфікація Shared-UV (v1)</div><div class="evo-track"><div class="evo-fill" style="width:76%"></div></div><div class="mono" style="text-align:right;">80.89%</div><div class="mono" style="text-align:right;color:var(--amber);">21.62%</div></div>
        <div class="evo-row"><div>3. CatBoost · 42 ознаки (v2_full)</div><div class="evo-track"><div class="evo-fill" style="width:84%"></div></div><div class="mono" style="text-align:right;">81.61%</div><div class="mono" style="text-align:right;color:var(--amber);">25.33%</div></div>
        <div class="evo-row"><div>4. Wave 1 E2E нейромережа (arm4)</div><div class="evo-track"><div class="evo-fill" style="width:87%"></div></div><div class="mono" style="text-align:right;">81.84%</div><div class="mono" style="text-align:right;color:var(--amber);">25.98%</div></div>
        <div class="evo-row"><div>5. Wave 2C · голова новинок (Option A)</div><div class="evo-track"><div class="evo-fill" style="width:90%"></div></div><div class="mono" style="text-align:right;">82.12%</div><div class="mono" style="text-align:right;color:var(--amber);">27.11%</div></div>
        <div class="evo-row"><div>6. 2× блоки пам'яті Transformer</div><div class="evo-track"><div class="evo-fill" style="width:91%"></div></div><div class="mono" style="text-align:right;">82.18%</div><div class="mono" style="text-align:right;color:var(--amber);">27.31%</div></div>
        <div class="evo-row best"><div>7. Фінал: 2× пам'ять + свіжість 365д ⭐</div><div class="evo-track"><div class="evo-fill" style="width:93%"></div></div><div class="mono" style="text-align:right;color:var(--green);">82.26%</div><div class="mono" style="text-align:right;color:var(--amber);">27.91%</div></div>
      </div>
    </div>
    <div class="grid-3">
      <div class="card green"><strong>+6.11 п.п. Recall@10</strong><p>Від 76.15% до 82.26% загальної точності Топ-10.</p></div>
      <div class="card amber"><strong>×2.5 краще відкриття новинок</strong><p>New@10 виріс із 11.25% до 27.91% (а в режимі чистих новинок — 52.27%).</p></div>
      <div class="card accent"><strong>Сильніша за CatBoost</strong><p>Нейромережа обійшла градієнтний бустинг за всіма основними метриками.</p></div>
    </div>
    <div class="slide-foot">
      <span>Кожен етап перевірявся на тій самій валідаційній вибірці (16 623 замовлення)</span>
      <span>Слайд 7 із 9</span>
    </div>
  </section>

  <!-- SLIDE 8 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Підсумкові результати · Валідація та Фінальний Тест</div>
      <h2>Точність фінальної моделі на валідації та незалежному тесті</h2>
      <p class="lead">
        Порівняння валідації (<strong>16 623 замовлення</strong>, 2026-02..06) та фінального тесту (<strong>13 184 замовлення</strong>, 2026-07..11 — через 5–10 місяців після навчання без жодного донавчання).
      </p>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Вибірка та сегмент</th>
            <th>Recall@10</th>
            <th>NDCG@10</th>
            <th>Нові@10</th>
            <th>Чисті новинки@10</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background:var(--green-bg);font-weight:700;">
            <td>Валідація · усі замовлення (16 623)</td>
            <td class="mono">82.26%</td>
            <td class="mono">65.48%</td>
            <td class="mono">27.91% (1 255)</td>
            <td class="mono">51.78% (2 328)</td>
          </tr>
          <tr style="background:var(--primary-soft);font-weight:700;">
            <td>Фінальний тест · модельований каталог (12 928)</td>
            <td class="mono">79.21%</td>
            <td class="mono">60.10%</td>
            <td class="mono">25.61% (853)</td>
            <td class="mono">50.47% (1 681)</td>
          </tr>
          <tr style="background:var(--amber-bg);font-weight:700;">
            <td>Фінальний тест · повний каталог (13 184)*</td>
            <td class="mono">76.63%</td>
            <td class="mono">58.41%</td>
            <td class="mono">21.71% (853)</td>
            <td class="mono">42.78% (1 681)</td>
          </tr>
          <tr>
            <td>Тест · постійні клієнти (12 867)</td>
            <td class="mono">77.61%</td>
            <td class="mono">59.32%</td>
            <td class="mono">18.74% (646)</td>
            <td class="mono">42.76% (1 474)</td>
          </tr>
          <tr>
            <td>Тест · точне фасування (Resolved SKU)</td>
            <td class="mono">54.15% (56.19% мод.)</td>
            <td class="mono">42.32%</td>
            <td class="mono">531 SKU</td>
            <td class="mono">—</td>
          </tr>
        </tbody>
      </table>
      <p style="font-size:12.5px;color:var(--muted);margin-top:8px;">
        *У повному каталозі тесту 831 позиція (3.1%) — це абсолютно нові продукти, що з'явились у прайсі вже після 2026-02-01 (поза замороженим каталогом). При регулярному квартальному перенавчанні вони автоматично входять у модель.
      </p>
    </div>
    <div class="slide-foot">
      <span>Усі метрики підтверджені у experiments/v2_full/v2_full_heldout_test_results.json</span>
      <span>Слайд 8 із 9</span>
    </div>
  </section>

  <!-- SLIDE 9 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Що входить у пакет</div>
      <h2>Готово до перегляду та інтеграції</h2>
      <p class="lead">
        У цьому ZIP-архіві зібрано все необхідне: інтерактивне демо на реальних кейсах валідації,
        презентація, зрозуміла документація архітектури та сама навчена модель PyTorch.
      </p>
    </div>
    <div class="grid-2">
      <div class="card accent">
        <h3>Файли для перегляду (відкриваються у браузері)</h3>
        <ul>
          <li><strong>OPEN_DEMO.html</strong> — інтерактивний перегляд прогнозів моделі на 220 реальних замовленнях із фінального тесту (2026-07..11).</li>
          <li><strong>PRESENTATION.html</strong> та <strong>PRESENTATION.pdf</strong> — ця презентація (слайд 5 має живу 3D-архітектуру).</li>
          <li><strong>DOCUMENTATION.html</strong> — зрозумілий посібник із інтерактивною 3D-моделлю архітектури та розбором усіх блоків.</li>
        </ul>
      </div>
      <div class="card">
        <h3>Навчена модель для розробників (model/)</h3>
        <ul>
          <li><strong>model/best_discovery.pt</strong> — фінальні ваги PyTorch (<code>__CKPT_MB__ МБ</code>, Епоха 10).</li>
          <li><strong>model/v2_full_reference.json</strong> — технічна специфікація та контрольна сума SHA-256.</li>
          <li class="mono" style="font-size:11.5px;word-break:break-all;color:var(--muted);">SHA-256: __CKPT_SHA__</li>
        </ul>
      </div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a class="btn primary" style="text-decoration:none;padding:11px 20px;font-size:14px;" href="OPEN_DEMO.html">Відкрити інтерактивне демо →</a>
      <a class="btn" style="text-decoration:none;padding:11px 20px;font-size:14px;" href="DOCUMENTATION.html">Читати документацію архітектури →</a>
    </div>
    <div class="slide-foot">
      <span>Дякую за увагу!</span>
      <span>Слайд 9 із 9</span>
    </div>
  </section>
</div>

<div class="dots" id="dots"></div>

<script>
const slides = Array.from(document.querySelectorAll('.slide'));
let idx = 0;

const dotsEl = document.getElementById('dots');
dotsEl.innerHTML = slides.map((_, i) => `<button class="dot" data-i="${i}">${i + 1}</button>`).join('');

function showSlide(n) {
  idx = Math.max(0, Math.min(slides.length - 1, n));
  slides.forEach((s, i) => s.classList.toggle('active', i === idx));
  document.querySelectorAll('.dot').forEach((d, i) => d.classList.toggle('active', i === idx));
  document.getElementById('slide-counter').textContent = `${idx + 1} / ${slides.length}`;
  document.getElementById('progress-fill').style.width = `${((idx + 1) / slides.length) * 100}%`;
}

document.getElementById('btn-prev').addEventListener('click', () => showSlide(idx - 1));
document.getElementById('btn-next').addEventListener('click', () => showSlide(idx + 1));
dotsEl.addEventListener('click', e => {
  if (e.target.matches('.dot')) showSlide(parseInt(e.target.dataset.i, 10));
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); showSlide(idx + 1); }
  if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); showSlide(idx - 1); }
  if (e.key === 'Home') showSlide(0);
  if (e.key === 'End') showSlide(slides.length - 1);
});

showSlide(0);

__ARCH_3D_JS__
mountArch3D('pres-arch3d', 'pres-3d-info', 'pres-3d-auto', 'pres-3d-reset');
</script>
</body>
</html>
"""


DOCUMENTATION_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Документація та архітектура — Система AI-рекомендацій B2B</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #2563eb;
    --primary-soft: rgba(37, 99, 235, 0.09);
    --green: #059669;
    --green-bg: rgba(16, 185, 129, 0.11);
    --green-border: #10b981;
    --amber: #b45309;
    --amber-bg: rgba(245, 158, 11, 0.13);
    --amber-border: #f59e0b;
    --purple: #6d28d9;
    --purple-bg: rgba(124, 58, 237, 0.10);
    --shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.65;
    padding: 18px 20px 60px;
  }
  .wrap { max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
  .nav-bar {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 18px; box-shadow: var(--shadow);
  }
  .nav-links { display: flex; flex-wrap: wrap; gap: 8px; }
  .nav-pill {
    text-decoration: none; font-size: 13px; font-weight: 600; color: var(--muted);
    padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface-2);
  }
  .nav-pill:hover { border-color: var(--primary); color: var(--primary); }
  .nav-pill.active { background: var(--primary); color: #fff; border-color: var(--primary); }
  .btn {
    border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
    font-family: var(--sans); font-size: 13px; font-weight: 600; padding: 6px 12px; border-radius: 8px; cursor: pointer;
  }
  .btn:hover { border-color: var(--primary); color: var(--primary); }

  .layout { display: grid; grid-template-columns: 250px 1fr; gap: 20px; align-items: start; }
  .toc {
    position: sticky; top: 18px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px; box-shadow: var(--shadow);
  }
  .toc h4 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 10px; }
  .toc a {
    display: block; text-decoration: none; color: #334155; font-size: 13.5px; font-weight: 600;
    padding: 6px 8px; border-radius: 7px; margin-bottom: 2px;
  }
  .toc a:hover { background: var(--primary-soft); color: var(--primary); }

  .doc { display: flex; flex-direction: column; gap: 20px; }
  .section {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 28px 32px; box-shadow: var(--shadow);
  }
  .section h1 { font-size: 30px; letter-spacing: -0.02em; line-height: 1.2; margin-bottom: 10px; }
  .section h2 { font-size: 21px; letter-spacing: -0.015em; margin-bottom: 12px; color: #0f172a; }
  .section h3 { font-size: 16px; font-weight: 800; margin: 16px 0 6px; color: #1e293b; }
  .section p { color: #334155; margin-bottom: 12px; }
  .section ul { padding-left: 20px; color: #334155; margin-bottom: 12px; display: flex; flex-direction: column; gap: 6px; }

  .callout {
    background: var(--primary-soft); border-left: 4px solid var(--primary);
    border-radius: 10px; padding: 14px 16px; margin: 14px 0; color: #1e3a8a; font-size: 14.5px;
  }
  .callout.green { background: var(--green-bg); border-left-color: var(--green-border); color: #065f46; }
  .callout.amber { background: var(--amber-bg); border-left-color: var(--amber-border); color: #78350f; }

  .arch-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
  .block-card {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
  }
  .block-num {
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11.5px; font-weight: 800; color: var(--primary); background: var(--primary-soft);
    padding: 3px 9px; border-radius: 999px; margin-bottom: 8px;
  }
  .block-card h3 { margin: 0 0 6px; font-size: 15.5px; }
  .block-card p { font-size: 13.8px; margin-bottom: 0; }

  .svg-frame {
    background: #fcfdff; border: 1px solid var(--border); border-radius: 14px; padding: 16px;
    margin: 14px 0; overflow-x: auto;
  }
  .svg-frame svg { width: 100%; height: auto; display: block; }

  table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 10px; }
  th { text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); padding: 9px 10px; border-bottom: 2px solid var(--border); }
  td { padding: 10px 10px; border-bottom: 1px solid var(--border); vertical-align: top; color: #334155; }
  .mono { font-family: var(--mono); font-size: 13px; }
  .pill {
    display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 9px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--surface-2);
  }
  .pill.hit { background: var(--green-bg); border-color: var(--green-border); color: var(--green); }
  .pill.newhit { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber); }

  @media (max-width: 900px) {
    .layout, .arch-grid { grid-template-columns: 1fr; }
    .toc { position: static; }
    .section { padding: 20px; }
  }
  @media print {
    body { background: #fff; padding: 0; }
    .nav-bar, .toc { display: none !important; }
    .layout { grid-template-columns: 1fr; }
    .section { box-shadow: none; break-inside: avoid; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="nav-bar">
    <div class="nav-links">
      <a class="nav-pill" href="OPEN_DEMO.html">Інтерактивне демо</a>
      <a class="nav-pill" href="PRESENTATION.html">Презентація</a>
      <a class="nav-pill active" href="DOCUMENTATION.html">Документація та архітектура</a>
    </div>
    <button class="btn" onclick="window.print()">Друк / Зберегти PDF</button>
  </div>

  <div class="layout">
    <aside class="toc">
      <h4>Зміст документації</h4>
      <a href="#intro">1. Що це за система</a>
      <a href="#data">2. Дані та чесна перевірка</a>
      <a href="#architecture">3. Архітектура моделі (схема)</a>
      <a href="#blocks">4. З чого складається кожен блок</a>
      <a href="#signals">5. 42 сигнали, на які дивиться AI</a>
      <a href="#recency">6. Навчання та вага свіжості</a>
      <a href="#demo-guide">7. Як читати демо і метрики</a>
      <a href="#passport">8. Технічний паспорт моделі</a>
    </aside>

    <main class="doc">
      <!-- 1. INTRO -->
      <section class="section" id="intro">
        <h1>Як працює система AI-рекомендацій B2B</h1>
        <p style="font-size:16px;color:var(--muted);">
          Зрозумілий посібник про те, як влаштована фінальна нейромережа <strong>w2_2mem_recency_hl365</strong>,
          з яких компонентів складається її архітектура і чому вона точно передбачає замовлення клієнтів.
        </p>
        <div class="callout green">
          <strong>Проста аналогія:</strong> уявіть дуже досвідченого менеджера з ідеальною пам'яттю.
          Перед дзвінком клієнту він за <strong>3 мілісекунди</strong> згадує всі минулі замовлення цього клієнта,
          його галузь, регіон, парк техніки, платіжну дисципліну та те, що зараз беруть схожі підприємства.
          Після цього він переглядає <strong>весь каталог із 1 141 товару</strong> і складає короткий список із
          <strong>Топ-10</strong> позицій, які клієнт найімовірніше купить сьогодні.
        </div>
        <p>
          Головна особливість цієї системи в тому, що вона одночасно вирішує <strong>три різні задачі</strong>:
        </p>
        <ul>
          <li><strong>Регулярні повтори:</strong> нагадує про товари, які клієнт уже брав раніше, саме тоді, коли підходить час поповнення.</li>
          <li><strong>Відкриття новинок (Крос-продажі):</strong> знаходить товари, яких клієнт <em>ще ніколи не купував</em>, але які природно доповнюють його асортимент.</li>
          <li><strong>Нові клієнти (1-ше замовлення):</strong> навіть без жодної минулої покупки формує точний стартовий список за профілем компанії та попитом у галузі.</li>
        </ul>
      </section>

      <!-- 2. DATA -->
      <section class="section" id="data">
        <h2>2. Які дані бачить модель і чому перевірка чесна</h2>
        <p>
          У реальному житті в момент прогнозу ми знаємо лише минуле і ніколи не знаємо майбутнього.
          Тому вся історія продажів розділена <strong>строго хронологічно</strong>:
        </p>
        <table>
          <thead>
            <tr><th>Період</th><th>Дати</th><th>Кількість замовлень</th><th>Роль у системі</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Навчання (Train)</strong></td>
              <td class="mono">&lt; 2026-02-01</td>
              <td class="mono">137 765</td>
              <td>На цих даних модель вивчала закономірності попиту та поведінки клієнтів.</td>
            </tr>
            <tr>
              <td><strong>Валідація (Validation)</strong></td>
              <td class="mono">2026-02-02 .. 2026-06-30</td>
              <td class="mono">16 623</td>
              <td>Підбір найкращої архітектури та калібрування (<code>Recall@10 = 82.26%</code>).</td>
            </tr>
            <tr style="background:var(--amber-bg);">
              <td><strong>Фінальний тест (Test · Демо)</strong></td>
              <td class="mono">2026-07-01 .. 2026-11-20</td>
              <td class="mono">13 184</td>
              <td>Незалежна фінальна перевірка через 5–10 міс. після навчання (<code>79.21%</code> мод. / <code>76.63%</code> повн.). Усі кейси в <code>OPEN_DEMO.html</code> взяті саме звідси.</td>
            </tr>
          </tbody>
        </table>
        <div class="callout">
          <strong>Принцип нульового витоку (Zero Future Leakage):</strong> для кожного замовлення на дату <em>D</em>
          історія клієнта, популярність товарів, статистика менеджера й фінансові показники перераховуються
          <strong>виключно за днями строго до дати D</strong>. Модель не має жодного доступу до того, що станеться того дня чи пізніше.
        </div>
        <h3>Навіщо каталог об'єднано у 1 141 продукт?</h3>
        <p>
          У прайс-листі є <strong>1 660 SKU</strong>, але багато з них — це одна й та сама формуляція в різній тарі
          (наприклад, <em>4л каністра</em>, <em>20л відро</em> та <em>209л бочка</em>). Якби нейромережа ранжувала сирі SKU,
          один популярний продукт міг би зайняти 3–4 місця в Топ-10. Тому модель спочатку обирає <strong>10 різних продуктів</strong>,
          а на фінальному кроці спеціальний модуль підбирає клієнту звичне для нього фасування.
        </p>
      </section>

      <!-- 3. INTERACTIVE 3D ARCHITECTURE -->
      <section class="section" id="architecture">
        <h2>3. Інтерактивна 3D-модель архітектури нейромережі</h2>
        <p>
          Модель називається <strong>Wave2BE2ERanker</strong> (одностадійна наскрізна нейромережа).
          «Одностадійна» означає, що вона <strong>не відкидає товари заздалегідь</strong> простішим фільтром — вона пропускає
          через повну увагу кожен із <strong>1 141</strong> доступних продуктів каталогу.
          <strong>Потягніть 3D-схему мишкою, щоб покрутити її з усіх боків, і клікніть по будь-якому блоку!</strong>
        </p>
        <div class="svg-frame" style="padding:10px;">
          <canvas id="doc-arch3d" style="width:100%;height:420px;display:block;cursor:grab;border-radius:10px;"></canvas>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-top:6px;">
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button class="btn" id="doc-3d-auto">Авто-обертання: Увімк.</button>
            <button class="btn" id="doc-3d-reset">Ізометрія (скинути)</button>
            <button class="btn" id="doc-3d-top">Вид зверху</button>
          </div>
          <span style="font-size:12.5px;color:var(--muted);">Перетягування = обертання · Колесо миші = масштаб · Клік = опис блоку</span>
        </div>
        <div class="callout" id="doc-3d-info" style="margin-top:12px;">
          <strong>Оберіть будь-який 3D-блок на схемі вище</strong>, щоб побачити його точну роль, розмірність тензора та пояснення простими словами.
        </div>
        <details style="margin-top:12px;">
          <summary style="cursor:pointer;font-weight:700;color:var(--primary);font-size:14px;">Показати також плоску 2D-схему (зручно для друку на папері)</summary>
          <div class="svg-frame">
          <svg viewBox="0 0 980 560" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Повна схема архітектури Wave2BE2ERanker">
            <defs>
              <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/>
              </marker>
              <marker id="arB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
              </marker>
            </defs>

            <!-- Step 1: Product Tower -->
            <rect x="20" y="18" width="290" height="122" rx="14" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.8"/>
            <text x="165" y="42" text-anchor="middle" font-size="13.5" font-weight="800" fill="#92400e">БЛОК 1 · Паспорт товарів (ProductTower)</text>
            <text x="165" y="64" text-anchor="middle" font-size="12" fill="#334155">Для кожного з 1 141 продукту каталогу:</text>
            <text x="165" y="84" text-anchor="middle" font-size="11.5" fill="#475569">• Текстовий зміст назви (BGE-M3 · 1024D)</text>
            <text x="165" y="102" text-anchor="middle" font-size="11.5" fill="#475569">• 65D специфікацій (тара, об'єм, галузь, в'язкість)</text>
            <text x="165" y="124" text-anchor="middle" font-size="12" font-weight="700" fill="#b45309">→ Компактний вектор товару v_bn (128 чисел)</text>

            <!-- Step 2: Customer Profile + History -->
            <rect x="345" y="18" width="290" height="122" rx="14" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.8"/>
            <text x="490" y="42" text-anchor="middle" font-size="13.5" font-weight="800" fill="#1e40af">БЛОК 2А · Профіль клієнта (Токен 0)</text>
            <text x="490" y="64" text-anchor="middle" font-size="12" fill="#334155">Хто цей клієнт на дату прогнозу:</text>
            <text x="490" y="84" text-anchor="middle" font-size="11.5" fill="#475569">• Текстовий опис компанії (BGE-M3)</text>
            <text x="490" y="102" text-anchor="middle" font-size="11.5" fill="#475569">• Місто, регіон, галузь діяльності</text>
            <text x="490" y="124" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">+ 7 ознак поведінки й фінансів (CUST_7)</text>

            <rect x="670" y="18" width="290" height="122" rx="14" fill="#f0fdf4" stroke="#10b981" stroke-width="1.8"/>
            <text x="815" y="42" text-anchor="middle" font-size="13.5" font-weight="800" fill="#065f46">БЛОК 2Б · Історія покупок (Токени 1..32)</text>
            <text x="815" y="64" text-anchor="middle" font-size="12" fill="#334155">Останні 32 замовлені продукти (&lt; дати):</text>
            <text x="815" y="84" text-anchor="middle" font-size="11.5" fill="#475569">• Вектор кожного купленого товару (v_bn)</text>
            <text x="815" y="102" text-anchor="middle" font-size="11.5" fill="#475569">• 5 часових ознак (SEQ_5: давність, об'єм)</text>
            <text x="815" y="124" text-anchor="middle" font-size="12" font-weight="700" fill="#047857">→ Єдина стрічка пам'яті: 33 токени × 128D</text>

            <!-- Arrows down to Memory Transformer -->
            <line x1="490" y1="140" x2="590" y2="176" stroke="#64748b" stroke-width="1.8" marker-end="url(#ar)"/>
            <line x1="815" y1="140" x2="710" y2="176" stroke="#64748b" stroke-width="1.8" marker-end="url(#ar)"/>

            <!-- Step 3: 2-Layer Memory Transformer -->
            <rect x="375" y="180" width="560" height="92" rx="14" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
            <text x="655" y="206" text-anchor="middle" font-size="14" font-weight="800" fill="#312e81">БЛОК 3 · 2-шаровий Transformer пам'яті клієнта</text>
            <rect x="405" y="220" width="235" height="36" rx="8" fill="#ffffff" stroke="#a5b4fc"/>
            <text x="522" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Шар 1: Self-Attention → FFN(384)</text>
            <line x1="640" y1="238" x2="670" y2="238" stroke="#6366f1" stroke-width="2" marker-end="url(#ar)"/>
            <rect x="674" y="220" width="235" height="36" rx="8" fill="#ffffff" stroke="#a5b4fc"/>
            <text x="791" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Шар 2: Self-Attention → FFN(384)</text>

            <!-- Step 4: Cross-Attention -->
            <line x1="165" y1="140" x2="290" y2="314" stroke="#2563eb" stroke-width="2" marker-end="url(#arB)"/>
            <line x1="655" y1="272" x2="540" y2="314" stroke="#2563eb" stroke-width="2" marker-end="url(#arB)"/>

            <rect x="90" y="318" width="800" height="86" rx="14" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
            <text x="490" y="344" text-anchor="middle" font-size="14.5" font-weight="800" fill="#1e40af">БЛОК 4 · 8-голова перехресна увага (Cross-Attention) + 42 сигнали кандидатів</text>
            <text x="490" y="367" text-anchor="middle" font-size="12.5" fill="#334155">
              Кожен із 1 141 товарів (Запит Q = v_bn) «звертається» до 33 токенів пам'яті клієнта.
            </text>
            <text x="490" y="388" text-anchor="middle" font-size="12" font-weight="700" fill="#2563eb">
              42 каузальні сигнали підказують кожній із 8 голів уваги, на які саме минулі події та характеристики дивитися.
            </text>

            <!-- Arrows to Dual Heads -->
            <line x1="320" y1="404" x2="260" y2="438" stroke="#059669" stroke-width="2" marker-end="url(#ar)"/>
            <line x1="660" y1="404" x2="720" y2="438" stroke="#d97706" stroke-width="2" marker-end="url(#ar)"/>

            <!-- Step 5: Dual Heads -->
            <rect x="40" y="442" width="420" height="96" rx="14" fill="#ecfdf5" stroke="#10b981" stroke-width="2"/>
            <text x="250" y="468" text-anchor="middle" font-size="14" font-weight="800" fill="#065f46">БЛОК 5А · Main Head (Повтори)</text>
            <text x="250" y="489" text-anchor="middle" font-size="12" fill="#334155">Глибока мережа MLP (384 → 192 → 96 → 1)</text>
            <text x="250" y="509" text-anchor="middle" font-size="12" fill="#047857">Оцінює ймовірність регулярного поповнення знайомих товарів</text>
            <text x="250" y="527" text-anchor="middle" font-size="11.5" fill="#64748b">Вихід: оцінка повтору s_main</text>

            <rect x="520" y="442" width="420" height="96" rx="14" fill="#fffbeb" stroke="#f59e0b" stroke-width="2"/>
            <text x="730" y="468" text-anchor="middle" font-size="14" font-weight="800" fill="#92400e">БЛОК 5Б · Option A Head (Нові товари)</text>
            <text x="730" y="489" text-anchor="middle" font-size="12" fill="#334155">Спеціалізована голова уваги + MLP (0% dropout)</text>
            <text x="730" y="509" text-anchor="middle" font-size="12" fill="#b45309">Навчається виключно на товарах, які клієнт купує ВПЕРШЕ</text>
            <text x="730" y="527" text-anchor="middle" font-size="11.5" fill="#64748b">Вихід: оцінка новинки s_unseen → калібрування a·s_unseen + b</text>
          </svg>
        </div>
        </details>
      </section>

      <!-- 4. EXPLANATION OF EACH BLOCK -->
      <section class="section" id="blocks">
        <h2>4. Пояснення кожного блоку архітектури простими словами</h2>
        <div class="arch-grid">
          <div class="block-card">
            <div class="block-num">Блок 1</div>
            <h3>Паспорт товарів (ProductTower)</h3>
            <p>
              Перетворює назву продукту та його технічні параметри (тара, об'єм, сфера застосування, в'язкість)
              на компактний вектор із 128 чисел. Схожі за властивостями продукти опиняються поруч у цьому просторі.
              У роботі ці 1 141 векторів кешуються заздалегідь, тому прогноз працює миттєво.
            </p>
          </div>
          <div class="block-card">
            <div class="block-num">Блок 2</div>
            <h3>Профіль клієнта + Історія (33 токени)</h3>
            <p>
              Об'єднує «хто це» (опис компанії, місто, регіон, галузь і 7 фінансово-поведінкових метрик)
              із «що він купував» (до 32 останніх покупок із давністю та кількістю). Для нового клієнта з 0 покупок
              працює Токен 0 (профіль компанії).
            </p>
          </div>
          <div class="block-card">
            <div class="block-num">Блок 3</div>
            <h3>2-шаровий Transformer пам'яті</h3>
            <p>
              Два послідовні шари самостійної уваги (Self-Attention + FFN). Вони дозволяють моделі помічати
              складні патерни: наприклад, клієнт перейшов з одного бренду на інший, або купує моторну оливу
              разом із гідравлічною кожні 45 днів. Додавання другого шару покращило розуміння довгих залежностей.
            </p>
          </div>
          <div class="block-card">
            <div class="block-num">Блок 4</div>
            <h3>8-голова перехресна увага (Cross-Attention)</h3>
            <p>
              «Серце» моделі. Кожен із 1 141 товарів ставить 8 різних «запитань» до пам'яті клієнта одночасно:
              одна голова перевіряє збіг бренду, інша — циклічність повторів, третя — схожість із галуззю,
              четверта — зв'язки товарів. 42 сигнали керують силою цієї уваги.
            </p>
          </div>
          <div class="block-card">
            <div class="block-num">Блок 5</div>
            <h3>Дві спеціалізовані голови + Калібрування</h3>
            <p>
              Якби була лише одна голова, прості повтори завжди б витісняли новинки. Тому створено дві окремі голови:
              <strong>Main Head</strong> відповідає за точність повторів, а <strong>Option A Head</strong> спеціально
              тренується знаходити нові товари. Потім лінійне калібрування (<code>a·s_new + b</code>) зводить обидві шкали в один чесний Топ-10.
            </p>
          </div>
          <div class="block-card">
            <div class="block-num">Блок 6</div>
            <h3>Підбір фасування (SKU Resolver)</h3>
            <p>
              Коли Топ-10 продуктів уже знайдено, окремий каузальний алгоритм підбирає конкретний SKU:
              спочатку перевіряє, яку тару саме цей клієнт брав раніше, а якщо продукт новий — орієнтується
              на типовий об'єм клієнта (бочка, відро чи каністра) та популярність фасування в каталозі.
            </p>
          </div>
        </div>
      </section>

      <!-- 5. 42 SIGNALS -->
      <section class="section" id="signals">
        <h2>5. На які 42 сигнали спирається модель</h2>
        <p>
          Для кожної пари «Клієнт — Продукт» на дату замовлення автоматично обчислюється 42 числові підказки,
          згруповані у 6 зрозумілих бізнес-категорій:
        </p>
        <table>
          <thead>
            <tr><th>Група сигналів</th><th>Що саме вимірюється</th><th>Яку роль відіграє</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>1. Повтори і терміновість</strong></td>
              <td>Кількість минулих покупок товару, частка в замовленнях, днів з останньої покупки, відношення до звичного інтервалу клієнта.</td>
              <td>Головний сигнал для регулярного поповнення запасів.</td>
            </tr>
            <tr>
              <td><strong>2. Схожі клієнти (CF)</strong></td>
              <td>Колаборативна фільтрація: що купують інші підприємства з подібним кошиком покупок (косинусна схожість і свіжість).</td>
              <td>Найсильніший сигнал для відкриття нових товарів (крос-продажів).</td>
            </tr>
            <tr>
              <td><strong>3. Зв'язки товарів</strong></td>
              <td>Статистичний ліфт (Sequence Lift): як часто продукт B замовляють після продукту A.</td>
              <td>Знаходить природні комплекти (наприклад, моторна + трансмісійна олива).</td>
            </tr>
            <tr>
              <td><strong>4. Менеджер і регіон</strong></td>
              <td>Популярність товару у менеджера клієнта за 30 та 90 днів, перетин «Менеджер × Галузь», регіональний попит за 90 днів.</td>
              <td>Вловлює локальні акції, сезонність та реальну наявність на складах регіону.</td>
            </tr>
            <tr>
              <td><strong>5. Галузь, техніка, в'язкість</strong></td>
              <td>Попит у галузі клієнта, збіг домену обладнання (сільгосптехніка, вантажівки, промисловість), збіг класів в'язкості.</td>
              <td>Критично важливо для нових клієнтів без історії покупок.</td>
            </tr>
            <tr>
              <td><strong>6. Фінанси та масштаб</strong></td>
              <td>7 ознак клієнта (CUST_7): активність, різноманітність кошика, частка передплат, середня відстрочка, прострочення.</td>
              <td>Відрізняє великих корпоративних клієнтів від дрібних роздрібних покупців.</td>
            </tr>
          </tbody>
        </table>
      </section>

      <!-- 6. RECENCY WEIGHTING -->
      <section class="section" id="recency">
        <h2>6. Як навчалася модель: вага свіжості (365 днів)</h2>
        <p>
          У B2B-торгівлі ринок змінюється щороку: з'являються нові бренди, змінюються постачальники та вподобання клієнтів.
          Тому під час навчання фінальної моделі <strong>w2_2mem_recency_hl365</strong> кожному історичному замовленню
          присвоювалася експоненційна вага свіжості з періодом напіврозпаду <strong>365 днів</strong>:
        </p>
        <div class="svg-frame">
          <svg viewBox="0 0 820 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Графік експоненційної ваги свіжості HL=365 днів">
            <line x1="70" y1="150" x2="770" y2="150" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="70" y1="22" x2="70" y2="150" stroke="#94a3b8" stroke-width="1.5"/>
            <text x="58" y="34" text-anchor="end" font-size="11.5" fill="#64748b">1.0</text>
            <text x="58" y="92" text-anchor="end" font-size="11.5" fill="#64748b">0.5</text>
            <text x="58" y="140" text-anchor="end" font-size="11.5" fill="#64748b">0.1</text>

            <line x1="70" y1="88" x2="770" y2="88" stroke="#e2e8f0" stroke-dasharray="4,4"/>
            <line x1="70" y1="138" x2="770" y2="138" stroke="#e2e8f0" stroke-dasharray="4,4"/>

            <path d="M 70 138 C 210 136, 360 124, 495 88 C 595 62, 690 36, 760 28" fill="none" stroke="#2563eb" stroke-width="3.2"/>
            <circle cx="760" cy="28" r="6" fill="#059669"/>
            <circle cx="495" cy="88" r="5" fill="#2563eb"/>
            <circle cx="280" cy="126" r="4.5" fill="#64748b"/>
            <circle cx="90" cy="138" r="4.5" fill="#94a3b8"/>

            <text x="760" y="172" text-anchor="end" font-size="11.5" font-weight="700" fill="#059669">2026 (вага 1.0)</text>
            <text x="495" y="172" text-anchor="middle" font-size="11.5" font-weight="700" fill="#2563eb">−1 рік (вага 0.50)</text>
            <text x="280" y="172" text-anchor="middle" font-size="11.5" fill="#64748b">−2 роки (вага 0.25)</text>
            <text x="125" y="172" text-anchor="middle" font-size="11.5" fill="#94a3b8">−3+ роки (мін. 0.10)</text>
          </svg>
        </div>
        <p>
          Завдяки цьому модель зберігає довгу історичну пам'ять, але приймає рішення так, щоб якнайкраще відповідати
          сучасному асортименту 2025–2026 років.
        </p>
        <div class="callout amber">
          <strong>Як модель працює, якщо навчання завершилося раніше за «сьогодні»?</strong>
          Вага свіжості <code>HL = 365 днів</code> рахується відносно кінця навчального періоду (<code>2026-02-01</code>).
          Але під час щоденної роботи вхідні <strong>42 сигнали та історія 32 покупок оновлюються в реальному часі</strong>
          до поточної дати — тому навіть через <strong>5–10 місяців без перенавчання</strong> на фінальному тесті
          (<code>2026-07..2026-11</code>) модель зберегла <strong>79.21% Recall@10</strong> на модельованому каталозі
          (<strong>76.63%</strong> на повному каталозі з урахуванням 831 абсолютно нового товару, що з'явився після лютого 2026).
          У продакшені достатньо раз на квартал запускати перенавчання з новим <code>T_cutoff = сьогодні</code>, щоб модель
          автоматично підхоплювала нові товари каталогу й оновлювала фокус свіжості.
        </div>
      </section>

      <!-- 7. HOW TO READ THE DEMO -->
      <section class="section" id="demo-guide">
        <h2>7. Як користуватися демо та розуміти показники</h2>
        <p>Відкрийте файл <a href="OPEN_DEMO.html" style="color:var(--primary);font-weight:700;">OPEN_DEMO.html</a> — на екрані показано реальні замовлення клієнтів із фінального тесту (липень–листопад 2026):</p>
        <ul>
          <li><span class="pill hit">✓ Вгадано · Повтор</span> — зелений рядок означає, що клієнт справді купив цей знайомий товар того дня, і модель поставила його в Топ-10.</li>
          <li><span class="pill newhit">★ Вгадано · Новий!</span> — жовтий рядок означає найцінніше влучання: клієнт купив цей товар <strong>вперше в житті</strong>, а нейромережа передбачила це заздалегідь.</li>
          <li><strong>Теги під назвою товару</strong> (наприклад, <code>Купував 4× · 32д тому</code>, <code>Схожі клієнти</code>, <code>Регіон 90д</code>, <code>Галузь</code>) пояснюють головні причини, чому модель підняла продукт уверх.</li>
          <li><strong>Перемикач режиму:</strong> «Змішаний (Повтори + Новинки)» показує повний робочий список, а «Лише нові товари (Крос-продажі)» приховує вже куплені позиції й показує чисті пропозиції для розширення асортименту клієнта.</li>
          <li><strong>Recall@10</strong> — який відсоток реально куплених клієнтом товарів потрапив у Топ-10 рекомендацій.</li>
        </ul>
      </section>

      <!-- 8. PASSPORT -->
      <section class="section" id="passport">
        <h2>8. Технічний паспорт фінальної моделі в пакеті</h2>
        <table>
          <tbody>
            <tr><td><strong>Ідентифікатор релізу</strong></td><td class="mono">__REF_ID__</td></tr>
            <tr><td><strong>Архітектура</strong></td><td class="mono">Wave2BE2ERanker (2× Memory Transformer + OptionAUnseenHead)</td></tr>
            <tr><td><strong>Кількість параметрів</strong></td><td class="mono">__TRAIN_PARAMS__ (навчання) / __REQ_PARAMS__ (на 1 запит)</td></tr>
            <tr><td><strong>Каталог продуктів</strong></td><td class="mono">1 141 канонічний продукт (1 660 SKU)</td></tr>
            <tr><td><strong>Валідаційні метрики (16 623 замовлення)</strong></td><td class="mono">Recall@10: 82.26% · NDCG@10: 65.48% · New@10: 27.91% · PureU@10: 51.78%</td></tr>
            <tr><td><strong>Фінальний тест (13 184 зам., 2026-07..11)</strong></td><td class="mono">Модельов.: R@10 79.21% · NDCG 60.10% · PureU@10 50.47% | Повн. кат.: R@10 76.63% · SKU 54.15%</td></tr>
            <tr><td><strong>Швидкість прогнозу (CPU)</strong></td><td class="mono">~2.9 мс на 1 клієнта (по всіх 1 141 продуктах)</td></tr>
            <tr><td><strong>Файл моделі у ZIP</strong></td><td class="mono">model/best_discovery.pt (__CKPT_MB__ МБ)</td></tr>
            <tr><td><strong>Контрольна сума SHA-256</strong></td><td class="mono" style="word-break:break-all;">__CKPT_SHA__</td></tr>
          </tbody>
        </table>
      </section>
    </main>
  </div>
</div>
<script>
__ARCH_3D_JS__
mountArch3D('doc-arch3d', 'doc-3d-info', 'doc-3d-auto', 'doc-3d-reset', 'doc-3d-top');
</script>
</body>
</html>
"""


ARCH_3D_JS = r"""
function mountArch3D(canvasId, infoId, autoBtnId, resetBtnId, topBtnId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const infoEl = document.getElementById(infoId);
  const autoBtn = document.getElementById(autoBtnId);
  const resetBtn = document.getElementById(resetBtnId);
  const topBtn = topBtnId ? document.getElementById(topBtnId) : null;

  const BLOCKS = [
    {
      id: 'cust', label: '1. Профіль клієнта', sub: 'Токен 0 · 1×128D',
      x: -7.2, y: 2.0, z: 0.1, dx: 1.7, dy: 1.5, dz: 1.15,
      top: '#60a5fa', front: '#3b82f6', side: '#1d4ed8',
      desc: '<strong>Профіль клієнта (Токен 0 · 1×128D):</strong> об’єднує текстовий опис компанії (BGE-M3), місто, регіон, галузь та 7 фінансово-поведінкових ознак (CUST_7: передплата, відстрочка, активність). Саме цей блок працює для нових клієнтів із 0 покупок.'
    },
    {
      id: 'hist', label: '2. Історія (32 покупки)', sub: 'Токени 1..32 · 32×128D',
      x: -7.2, y: 0.0, z: 0.1, dx: 1.7, dy: 1.45, dz: 1.25, stacks: 4,
      top: '#34d399', front: '#10b981', side: '#047857',
      desc: '<strong>Історія покупок (Токени 1..32 · 32×128D):</strong> останні 32 замовлені продукти клієнта строго до дати прогнозу. Кожна платівка несе вектор товару (v_bn) + 5 часових і кількісних ознак (SEQ_5).'
    },
    {
      id: 'cat', label: '3. Каталог ProductTower', sub: '1 141 × 128D',
      x: -7.2, y: -2.1, z: 0.1, dx: 1.7, dy: 1.5, dz: 1.25, stacks: 4,
      top: '#fbbf24', front: '#f59e0b', side: '#b45309',
      desc: '<strong>Паспорт каталогу (ProductTower · 1 141×128D):</strong> перетворює назву (BGE-M3) та 65 технічних характеристик (тара, об’єм, галузь, в’язкість) кожного продукту на вектор із 128 чисел. Кешується заздалегідь для швидкості 2.9 мс.'
    },
    {
      id: 'mem1', label: '4А. Transformer · Шар 1', sub: '33×128D → FFN 384',
      x: -3.8, y: 0.7, z: 0.1, dx: 1.25, dy: 2.7, dz: 1.9,
      top: '#a5b4fc', front: '#6366f1', side: '#4338ca',
      desc: '<strong>Пам’ять клієнта · Transformer Шар 1:</strong> аналізує зв’язки між профілем клієнта та його 32 минулими покупками (Self-Attention + FFN 384), виявляючи цикли поповнення та комплекти товарів.'
    },
    {
      id: 'mem2', label: '4Б. Transformer · Шар 2', sub: '33×128D → FFN 384',
      x: -2.15, y: 0.7, z: 0.1, dx: 1.25, dy: 2.7, dz: 1.9,
      top: '#818cf8', front: '#4f46e5', side: '#312e81',
      desc: '<strong>Пам’ять клієнта · Transformer Шар 2:</strong> другий глибокий шар пам’яті формує узагальнене уявлення про звички клієнта, переходи між брендами та зміну потреб у часі.'
    },
    {
      id: 'sig42', label: '5. 42 каузальні сигнали', sub: '1 141 × 42D (< дати)',
      x: -3.5, y: -2.1, z: 0.1, dx: 2.5, dy: 1.55, dz: 1.15,
      top: '#38bdf8', front: '#0284c7', side: '#075985',
      desc: '<strong>42 каузальні сигнали (1 141×42D):</strong> історія повторів, схожі клієнти (CF), зв’язки товарів, попит менеджера й регіону за 30/90 днів, галузь і в’язкість — усе обчислюється строго до дати замовлення.'
    },
    {
      id: 'xatt', label: '6. 8-голова Cross-Attention', sub: '8 голів × Full-42 Bias',
      x: 0.5, y: -0.7, z: 0.1, dx: 2.25, dy: 3.4, dz: 2.25, stacks: 8,
      top: '#60a5fa', front: '#2563eb', side: '#1e3a8a',
      desc: '<strong>8-голова перехресна увага (Cross-Attention + Full-42 Bias):</strong> кожен із 1 141 товарів «запитує» пам’ять клієнта з 8 різних ракурсів одночасно, підсилюючи увагу за допомогою всіх 42 бізнес-сигналів.'
    },
    {
      id: 'main', label: '7А. Main Head (Повтори)', sub: 'MLP 384→192→96→1',
      x: 4.1, y: 1.25, z: 0.1, dx: 1.65, dy: 1.85, dz: 1.55,
      top: '#34d399', front: '#10b981', side: '#047857',
      desc: '<strong>Голова 1 · Main Head (Регулярні повтори):</strong> глибокий MLP, навчений із вагою свіжості HL=365 днів. Точно визначає, які знайомі товари клієнт має докупити саме сьогодні.'
    },
    {
      id: 'opta', label: '7Б. Option A (Новинки)', sub: '0% dropout · Cross-Sell',
      x: 4.1, y: -1.45, z: 0.1, dx: 1.65, dy: 1.85, dz: 1.55,
      top: '#fcd34d', front: '#f59e0b', side: '#b45309',
      desc: '<strong>Голова 2 · Option A Head (Нові товари):</strong> окрема спеціалізована голова, яка тренується виключно на товарах, що клієнт купує ВПЕРШЕ. Дає 50.47% точність чистих крос-продажів на фінальному тесті.'
    },
    {
      id: 'out', label: '8. Калібрування + SKU', sub: 'a·s_new + b → Топ-10',
      x: 6.75, y: -0.25, z: 0.15, dx: 1.45, dy: 2.35, dz: 2.05,
      top: '#3b82f6', front: '#1d4ed8', side: '#1e3a8a',
      desc: '<strong>Калібрування + Підбір тари (SKU Resolver):</strong> зводить оцінки двох голів у єдиний Топ-10 продуктів та автоматично підбирає конкретне фасування (бочка 209л, відро 20л, каністра 4л).'
    },
  ];

  const EDGES = [
    ['cust', 'mem1', '#2563eb'], ['hist', 'mem1', '#10b981'],
    ['mem1', 'mem2', '#4f46e5'], ['mem2', 'xatt', '#4f46e5'],
    ['cat', 'sig42', '#d97706'], ['sig42', 'xatt', '#0284c7'],
    ['xatt', 'main', '#10b981'], ['xatt', 'opta', '#f59e0b'],
    ['main', 'out', '#10b981'], ['opta', 'out', '#f59e0b'],
  ];

  const byId = Object.fromEntries(BLOCKS.map(b => [b.id, b]));
  let yaw = -0.46, pitch = 0.44, zoom = 1.0, autoSpin = true, selectedId = 'xatt';
  let dragging = false, lx = 0, ly = 0, draggedDist = 0;
  let hitZones = [];
  let tAnim = 0;

  if (infoEl && byId[selectedId]) infoEl.innerHTML = byId[selectedId].desc;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  function proj(x, y, z, W, H) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = x * cy - y * sy;
    const y1 = x * sy + y * cy;
    const y2 = y1 * cp - z * sp;
    const z2 = y1 * sp + z * cp;
    const dist = 23.0;
    const persp = dist / (dist - y2);
    const scale = Math.min(W / 18.2, H / 7.8) * zoom * persp;
    return { sx: W * 0.5 + x1 * scale, sy: H * 0.55 - z2 * scale, depth: y2, scale };
  }

  function centerOf(b) {
    return [b.x + b.dx * 0.5, b.y + b.dy * 0.5, b.z + b.dz * 0.5];
  }

  function drawPrism(x, y, z, dx, dy, dz, topC, frontC, sideC, isSel, W, H) {
    const pts = [
      [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
      [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ].map(p => proj(p[0], p[1], p[2], W, H));

    const faces = [
      { idx: [4,5,6,7], col: topC },
      { idx: [0,1,5,4], col: frontC },
      { idx: [1,2,6,5], col: sideC },
      { idx: [2,3,7,6], col: frontC },
      { idx: [3,0,4,7], col: sideC },
    ];
    faces.sort((a, b) => {
      const da = a.idx.reduce((s, i) => s + pts[i].depth, 0) / 4;
      const db = b.idx.reduce((s, i) => s + pts[i].depth, 0) / 4;
      return da - db;
    });
    faces.forEach(f => {
      ctx.beginPath();
      f.idx.forEach((vi, k) => k === 0 ? ctx.moveTo(pts[vi].sx, pts[vi].sy) : ctx.lineTo(pts[vi].sx, pts[vi].sy));
      ctx.closePath();
      ctx.fillStyle = f.col;
      ctx.globalAlpha = isSel ? 1.0 : 0.92;
      ctx.fill();
      ctx.strokeStyle = isSel ? '#0f172a' : 'rgba(255,255,255,0.85)';
      ctx.lineWidth = isSel ? 1.6 : 0.9;
      ctx.stroke();
    });
    ctx.globalAlpha = 1.0;
    const xs = pts.map(p => p.sx), ys = pts.map(p => p.sy);
    return {
      minX: Math.min(...xs), maxX: Math.max(...xs),
      minY: Math.min(...ys), maxY: Math.max(...ys),
      topCx: (pts[4].sx + pts[5].sx + pts[6].sx + pts[7].sx) / 4,
      topCy: (pts[4].sy + pts[5].sy + pts[6].sy + pts[7].sy) / 4,
    };
  }

  function frame() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      if (Math.abs(canvas.width - Math.round(rect.width * Math.min(window.devicePixelRatio || 1, 2))) > 2) resize();
      const W = rect.width, H = rect.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#f8fbff';
      ctx.fillRect(0, 0, W, H);

      if (autoSpin && !dragging) yaw = -0.46 + Math.sin(tAnim * 0.45) * 0.24;
      tAnim += 0.016;

      // Subtle 3D floor grid
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 0.8;
      for (let gx = -8; gx <= 9; gx += 1.5) {
        const a = proj(gx, -3.2, 0, W, H), b = proj(gx, 3.8, 0, W, H);
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
      }
      for (let gy = -3; gy <= 3.6; gy += 1.1) {
        const a = proj(-8, gy, 0, W, H), b = proj(9, gy, 0, W, H);
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
      }

      // Draw 3D flow lines & animated data pulses
      EDGES.forEach(([u, v, col], ei) => {
        const p1 = centerOf(byId[u]), p2 = centerOf(byId[v]);
        const s1 = proj(p1[0], p1[1], p1[2], W, H);
        const s2 = proj(p2[0], p2[1], p2[2], W, H);
        ctx.beginPath();
        ctx.moveTo(s1.sx, s1.sy);
        ctx.lineTo(s2.sx, s2.sy);
        ctx.strokeStyle = col;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 2.0;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        const frac = (tAnim * 0.55 + ei * 0.13) % 1.0;
        const px = p1[0] + (p2[0] - p1[0]) * frac;
        const py = p1[1] + (p2[1] - p1[1]) * frac;
        const pz = p1[2] + (p2[2] - p1[2]) * frac;
        const sp = proj(px, py, pz, W, H);
        ctx.beginPath();
        ctx.arc(sp.sx, sp.sy, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
      });

      // Depth-sort blocks back to front
      const sorted = BLOCKS.map(b => {
        const c = centerOf(b);
        return { b, depth: proj(c[0], c[1], c[2], W, H).depth };
      }).sort((a, b) => a.depth - b.depth);

      hitZones = [];
      const labels = [];
      sorted.forEach(({ b }) => {
        const isSel = b.id === selectedId;
        let bbox;
        if (b.stacks && b.stacks > 1) {
          const gap = 0.06;
          const sh = (b.dz - gap * (b.stacks - 1)) / b.stacks;
          for (let s = 0; s < b.stacks; s++) {
            const z0 = b.z + s * (sh + gap);
            bbox = drawPrism(b.x, b.y, z0, b.dx, b.dy, sh, b.top, b.front, b.side, isSel, W, H);
          }
        } else {
          bbox = drawPrism(b.x, b.y, b.z, b.dx, b.dy, b.dz, b.top, b.front, b.side, isSel, W, H);
        }
        hitZones.push({ id: b.id, ...bbox });
        labels.push({ b, isSel, x: bbox.topCx, y: bbox.minY - 8 });
      });

      // Crisp floating Ukrainian pill labels
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      labels.forEach(({ b, isSel, x, y }) => {
        ctx.font = isSel ? '700 11.5px -apple-system, sans-serif' : '600 10.5px -apple-system, sans-serif';
        const tw = ctx.measureText(b.label).width + 14;
        const th = 20;
        const lx0 = Math.max(8, Math.min(W - tw - 8, x - tw / 2));
        const ly0 = Math.max(6, Math.min(H - th - 6, y - th / 2));
        ctx.fillStyle = isSel ? '#0f172a' : 'rgba(255,255,255,0.92)';
        ctx.strokeStyle = isSel ? '#0f172a' : '#cbd5e1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(lx0, ly0, tw, th, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = isSel ? '#ffffff' : '#0f172a';
        ctx.fillText(b.label, lx0 + tw / 2, ly0 + th / 2 + 0.5);
      });
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  canvas.addEventListener('pointerdown', e => {
    dragging = true; draggedDist = 0; lx = e.clientX; ly = e.clientY;
    canvas.style.cursor = 'grabbing';
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    draggedDist += Math.hypot(dx, dy);
    if (draggedDist > 4) {
      autoSpin = false;
      if (autoBtn) autoBtn.textContent = 'Авто-обертання: Вимк.';
    }
    yaw += dx * 0.008;
    pitch = Math.max(0.08, Math.min(1.28, pitch + dy * 0.008));
  });
  canvas.addEventListener('pointerup', e => {
    dragging = false;
    canvas.style.cursor = 'grab';
    if (draggedDist < 6) {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      for (let i = hitZones.length - 1; i >= 0; i--) {
        const z = hitZones[i];
        if (mx >= z.minX - 6 && mx <= z.maxX + 6 && my >= z.minY - 24 && my <= z.maxY + 6) {
          selectedId = z.id;
          if (infoEl) infoEl.innerHTML = byId[z.id].desc;
          break;
        }
      }
    }
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    zoom = Math.max(0.65, Math.min(1.65, zoom * (e.deltaY > 0 ? 0.94 : 1.06)));
  }, { passive: false });

  if (autoBtn) autoBtn.addEventListener('click', () => {
    autoSpin = !autoSpin;
    autoBtn.textContent = autoSpin ? 'Авто-обертання: Увімк.' : 'Авто-обертання: Вимк.';
  });
  if (resetBtn) resetBtn.addEventListener('click', () => {
    yaw = -0.46; pitch = 0.44; zoom = 1.0; autoSpin = true;
    if (autoBtn) autoBtn.textContent = 'Авто-обертання: Увімк.';
  });
  if (topBtn) topBtn.addEventListener('click', () => {
    yaw = 0.0; pitch = 1.24; zoom = 1.02; autoSpin = false;
    if (autoBtn) autoBtn.textContent = 'Авто-обертання: Вимк.';
  });
}
"""


def render_presentation_html(meta: Mapping[str, object]) -> str:
    ckpt_mb = f"{int(meta['checkpoint_bytes']) / (1024 * 1024):.2f}"
    return (
        PRESENTATION_HTML_TEMPLATE.replace("__CKPT_MB__", ckpt_mb)
        .replace("__CKPT_SHA__", str(meta["checkpoint_sha256"]))
        .replace("__ARCH_3D_JS__", ARCH_3D_JS)
    )


def render_documentation_html(meta: Mapping[str, object]) -> str:
    ckpt_mb = f"{int(meta['checkpoint_bytes']) / (1024 * 1024):.2f}"
    return (
        DOCUMENTATION_HTML_TEMPLATE.replace("__REF_ID__", str(meta["reference_id"]))
        .replace("__TRAIN_PARAMS__", f"{int(meta['train_params']):,}".replace(",", " "))
        .replace("__REQ_PARAMS__", f"{int(meta['per_request_params']):,}".replace(",", " "))
        .replace("__CKPT_MB__", ckpt_mb)
        .replace("__CKPT_SHA__", str(meta["checkpoint_sha256"]))
        .replace("__ARCH_3D_JS__", ARCH_3D_JS)
    )
