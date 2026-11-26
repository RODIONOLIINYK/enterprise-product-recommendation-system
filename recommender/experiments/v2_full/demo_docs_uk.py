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
        <div class="kpi-lbl">Валідація · Recall@10</div>
        <div class="kpi-big green">84.27%</div>
        <div class="kpi-sub">Навчені C&amp;P (82.26% усі 16 623 зам.)</div>
      </div>
      <div class="card accent">
        <div class="kpi-lbl">Фінальний тест · Recall@10</div>
        <div class="kpi-big blue">83.03%</div>
        <div class="kpi-sub">Навчені C&amp;P (76.63% усі 13 184)</div>
      </div>
      <div class="card amber">
        <div class="kpi-lbl">Якість порядку (NDCG@10)</div>
        <div class="kpi-big amber">67.18%</div>
        <div class="kpi-sub">Val навч. C&amp;P (63.49% тест / 58.41% усі)</div>
      </div>
      <div class="card">
        <div class="kpi-lbl">Швидкість роботи (CPU)</div>
        <div class="kpi-big">2.9 мс</div>
        <div class="sub kpi-sub">Без GPU · 1.13M параметрів</div>
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
        <div class="kpi-lbl" style="color:#047857;">2. Валідація (Val)</div>
        <div class="mono" style="font-size:20px;font-weight:800;margin:6px 0;">2026-02 → 2026-06</div>
        <p style="font-size:14px;color:#334155;"><strong>15 343 замовлення</strong> (навчені клієнти й товари) — <strong>84.27% Recall@10</strong>. Використовувалась для вибору архітектури.</p>
      </div>
      <div class="t-seg t-test" style="background:#fffbeb;border-color:#f59e0b;border-style:solid;">
        <div class="kpi-lbl" style="color:#b45309;">3. Фінальний тест (Демо)</div>
        <div class="mono" style="font-size:20px;font-weight:800;margin:6px 0;">2026-07 → 11-20</div>
        <p style="font-size:14px;color:#334155;"><strong>9 839 замовлень</strong> (навчені клієнти й товари, через 5–10 міс. після навчання) — <strong>83.03% Recall@10</strong>. Саме звідси взяті кейси в <code>OPEN_DEMO.html</code>.</p>
      </div>
    </div>
    <div class="card">
      <strong>Жодного перенавчання:</strong> між валідацією (<strong>84.27%</strong>) і тестом через 5–10 місяців (<strong>83.03%</strong>) різниця лише 1.24 п.п., і якість стабільна кожен місяць від липня (82.33%) до листопада (82.88%).
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
          <li><strong>Свіжі продажі (січень 2026):</strong> ефективна вага <strong>2.31×</strong> (понад 2× від середнього!)</li>
          <li><strong>1 рік тому (лютий 2025):</strong> ефективна вага <strong>1.16×</strong> (січень 2025: <strong>1.09×</strong>)</li>
          <li><strong>2 роки тому (лютий 2024):</strong> ефективна вага <strong>0.58×</strong></li>
          <li><strong>3 роки тому (січень 2023):</strong> ефективна вага <strong>0.28×</strong> (середня по вибірці = 1.00×)</li>
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
        <div class="evo-row"><div>1. Базова популярність (старт, <code>a0f5407</code>)</div><div class="evo-track"><div class="evo-fill" style="width:21%;background:#94a3b8;"></div></div><div class="mono" style="text-align:right;">17.88%</div><div class="mono" style="text-align:right;color:var(--muted);">—</div></div>
        <div class="evo-row"><div>2. Перша Two-Tower (Stage 1 v1, <code>a0f5407</code>)</div><div class="evo-track"><div class="evo-fill" style="width:55%"></div></div><div class="mono" style="text-align:right;">47.60%</div><div class="mono" style="text-align:right;color:var(--muted);">—</div></div>
        <div class="evo-row"><div>3. Перший ранкер UPS-CANR (Stage 2 v1)</div><div class="evo-track"><div class="evo-fill" style="width:76%"></div></div><div class="mono" style="text-align:right;">66.69%</div><div class="mono" style="text-align:right;color:var(--muted);">—</div></div>
        <div class="evo-row"><div>4. CatBoost v1 · 52 ознаки (GBDT-бенчмарк)</div><div class="evo-track"><div class="evo-fill" style="width:84%;background:#64748b;"></div></div><div class="mono" style="text-align:right;color:#475569;">75.82%</div><div class="mono" style="text-align:right;color:#475569;">10.88%</div></div>
        <div class="evo-row"><div>5. 4 прапорці знайомості + Stat-Attention (<code>4f3d326</code>)</div><div class="evo-track"><div class="evo-fill" style="width:86%"></div></div><div class="mono" style="text-align:right;">76.15%</div><div class="mono" style="text-align:right;color:var(--amber);">11.25%</div></div>
        <div class="evo-row"><div>6. Канонічні продукти (base_name) + Full-Pool (<code>e0681bd</code>)</div><div class="evo-track"><div class="evo-fill" style="width:91%"></div></div><div class="mono" style="text-align:right;">80.89%</div><div class="mono" style="text-align:right;color:var(--amber);">21.62%</div></div>
        <div class="evo-row"><div>7. CatBoost v2 · 42 ознаки + Unseen Products Head (GBDT)</div><div class="evo-track"><div class="evo-fill" style="width:92%;background:#64748b;"></div></div><div class="mono" style="text-align:right;color:#475569;">81.61%</div><div class="mono" style="text-align:right;color:#475569;">25.33%</div></div>
        <div class="evo-row"><div>8. E2E Wave2BE2ERanker + Unseen Products Head (<code>w2_2mem_blocks</code>)</div><div class="evo-track"><div class="evo-fill" style="width:94%"></div></div><div class="mono" style="text-align:right;">82.18%</div><div class="mono" style="text-align:right;color:var(--amber);">27.31%</div></div>
        <div class="evo-row best"><div>9. Фінал: 2× пам'ять + свіжість 365д ⭐</div><div class="evo-track"><div class="evo-fill" style="width:97%"></div></div><div class="mono" style="text-align:right;color:var(--green);">82.26% / 84.27%*</div><div class="mono" style="text-align:right;color:var(--amber);">27.91%</div></div>
      </div>
    </div>
    <div class="grid-3">
      <div class="card green"><strong>+64.4 п.п. від старту</strong><p>Від 17.88% (популярність) та 47.60% (перша Two-Tower) до 82.26% усіх / 84.27%* навчених C&amp;P.</p></div>
      <div class="card amber"><strong>×2.5 краще відкриття новинок</strong><p>New@10 виріс із 11.25% до 27.91% (у режимі чистих новинок — 51.78%), а свіжість 365д дала +5.98 п.п. на нових товарах тесту.</p></div>
      <div class="card accent"><strong>Сильніша за CatBoost</strong><p>Нейромережа обійшла градієнтний бустинг (81.61% R@10 / 0.6438 NDCG) за всіма метриками.</p></div>
    </div>
    <div class="slide-foot">
      <span>*84.27% — валідація на навчених C&amp;P (83.03% у тесті через 7.4 міс.). Повний лог: PROJECT_HISTORY.md</span>
      <span>Слайд 7 із 9</span>
    </div>
  </section>

  <!-- SLIDE 8 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Підсумкові результати · Обидві оцінки (Навчені C&amp;P + Повна вибірка)</div>
      <h2>Порівняння валідації та фінального тесту (2026-07..11)</h2>
      <p class="lead">
        Показано обидва зрізи: <strong>навчені клієнти й товари</strong> (чиста перевірка без перенавчання) та <strong>повна вибірка</strong> (включно з новими пост-навчальними клієнтами й товарами).
      </p>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Вибірка / Зріз оцінки</th>
            <th>Замовлень</th>
            <th>Recall@10</th>
            <th>NDCG@10</th>
            <th>Нові@10 (Чисті*)</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background:var(--green-bg);font-weight:700;">
            <td>Валідація · навчені клієнти й товари</td>
            <td class="mono">15 343</td>
            <td class="mono">84.27%</td>
            <td class="mono">67.18%</td>
            <td class="mono">20.52% (582)</td>
          </tr>
          <tr style="background:var(--primary-soft);font-weight:700;">
            <td>Фінальний тест · навчені клієнти й товари</td>
            <td class="mono">9 839</td>
            <td class="mono">83.03%</td>
            <td class="mono">63.49%</td>
            <td class="mono">15.69% (43.1% чист.)</td>
          </tr>
          <tr>
            <td>Валідація · усі замовлення (повний каталог)</td>
            <td class="mono">16 623</td>
            <td class="mono">82.26%</td>
            <td class="mono">65.48%</td>
            <td class="mono">27.91% (51.8% чист.)</td>
          </tr>
          <tr style="background:var(--amber-bg);">
            <td>Фінальний тест · усі замовлення (повний каталог)**</td>
            <td class="mono">13 184</td>
            <td class="mono">76.63%</td>
            <td class="mono">58.41%</td>
            <td class="mono">21.71% (42.8% чист.)</td>
          </tr>
        </tbody>
      </table>
      <p style="font-size:12px;color:var(--muted);margin-top:8px;">
        *Чисті новинки — режим Unseen Products Head. **У продакшені <code>ProductTower</code> одразу кодує нові товари через опис і 65D спец. без перенавчання; у тесті таблиця була заморожена на 1 141 продукті (76.63% — нижня оцінка).<br />
        <strong>Без перенавчання:</strong> розрив від останнього навчання (<code>2026-01-30</code>) до середнього тесту (<code>2026-09-12</code>) — <strong>7.4 міс. (225 дн., до 9.7 міс.)</strong>, а точність на навчених C&amp;P тримається на <strong>83.03%</strong>.<br />
        <strong>Дешеве оновлення в роботі:</strong> донавчання моделі <strong>раз на 2 місяці або після додавання ~10 нових товарів</strong> коштує копійки й постійно утримує якість біля рівня навчених клієнтів і товарів (~<strong>84–85% Recall@10</strong>, як <code>84.27%</code> на валідації).
      </p>
    </div>
    <div class="slide-foot">
      <span>Стабільність через 5–9.7 міс. (навчені C&amp;P): Лип 82.33% → Сер 83.07% → Вер 83.72% → Жов 83.20% → Лис 82.88%</span>
      <span>Слайд 8 із 9</span>
    </div>
  </section>

  <!-- SLIDE 9 -->
  <section class="slide">
    <div>
      <div class="slide-kicker">Що входить у пакет</div>
      <h2>Готово до перегляду, живого запуску Python та інтеграції</h2>
      <p class="lead">
        У цьому ZIP-архіві зібрано і живий Python-сервер із реальним PyTorch-бенчмарком швидкості (~3 мс на CPU),
        і офлайн-файли для перегляду без встановлення програм.
      </p>
    </div>
    <div class="grid-2">
      <div class="card accent">
        <h3>Запуск демо, презентації та 3D-документації</h3>
        <ul>
          <li><strong>START_LIVE_DEMO.command / .bat</strong> (<code>live_demo_server.py</code>) — живий Python-сервер із реальним PyTorch CPU-інференсом на <strong>1 000 тестових замовленнях</strong> (700 навчених + 300 ненавчених) і живим бенчмарком швидкості!</li>
          <li><strong>OPEN_DEMO.html</strong> — офлайн-версія без Python (220 кейсів: 117 навчених + 103 ненавчених із <code>⚠️</code>).</li>
          <li><strong>PRESENTATION.html / .pdf</strong> та <strong>DOCUMENTATION.html</strong> — презентація й інтерактивна 3D-архітектура.</li>
        </ul>
      </div>
      <div class="card">
        <h3>Навчена модель та тензори у папці model/</h3>
        <ul>
          <li><strong>model/best_discovery.pt</strong> — фінальні ваги PyTorch (<code>__CKPT_MB__ МБ</code>, Епоха 10).</li>
          <li><strong>model/test_live_bundle.pt.gz</strong> — реальні тензори для живого PyTorch-демо.</li>
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
      <a href="#deployment">7. Два варіанти впровадження</a>
      <a href="#demo-guide">8. Як читати демо і метрики</a>
      <a href="#passport">9. Технічний паспорт моделі</a>
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
              <td class="mono">15 343 <span style="color:var(--muted);font-size:12px;">(навч. C&amp;P)</span></td>
              <td>Вибір найкращої архітектури та калібрування (<strong>Recall@10 = 84.27%</strong>, NDCG = 67.18%).</td>
            </tr>
            <tr style="background:var(--amber-bg);">
              <td><strong>Фінальний тест (Test · Демо)</strong></td>
              <td class="mono">2026-07-01 .. 2026-11-20</td>
              <td class="mono">9 839 <span style="color:var(--muted);font-size:12px;">(навч. C&amp;P)</span></td>
              <td>Незалежна перевірка через 5–10 міс. після навчання (<strong>Recall@10 = 83.03%</strong>, NDCG = 63.49%). Усі кейси в <code>OPEN_DEMO.html</code> взяті саме звідси.</td>
            </tr>
          </tbody>
        </table>
        <div class="callout green">
          <strong>Доказ відсутності перенавчання (часовий розрив 7.4 місяця):</strong>
          останній продаж у навчальній вибірці датований <code>2026-01-30</code>, тоді як <strong>середня дата замовлення у фінальному тесті — <code>2026-09-12</code></strong>
          (розрив <strong>225 днів = 7.4 місяця</strong>; медіана — <strong>7.5 місяця</strong>, а фінальні замовлення <code>2026-11-20</code> віддалені на <strong>9.7 місяця / 294 дні</strong>).
          Попри майже 7.5–10 місяців без жодного перенавчання, точність на навчених клієнтах і товарах тримається стабільно на рівні
          <strong>83.03% Recall@10</strong> (проти 84.27% на валідації) і не просідає помісячно (Лип 82.33% → Сер 83.07% → Вер 83.72% → Жов 83.20% → Лис 82.88%).
        </div>
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
            <text x="730" y="468" text-anchor="middle" font-size="14" font-weight="800" fill="#92400e">БЛОК 5Б · Unseen Products Head (Нові товари)</text>
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
              <strong>Main Head</strong> відповідає за точність повторів, а <strong>Unseen Products Head</strong> спеціально
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
          <strong>Ефективна вага свіжості та робота з новими товарами:</strong>
          Після нормалізації середньої ваги до <code>1.00×</code> найсвіжіші продажі (<code>2026-01</code>) отримують
          <strong>2.31× важливість</strong> (у понад 2 рази більше за середнє і у 8× більше за 2023 рік), рік тому
          (<code>2025-02</code>) — <strong>1.16×</strong> (січень 2025: <strong>1.09×</strong>), 2 роки тому — <strong>0.58×</strong>,
          3 роки тому — <strong>0.28×</strong>. Під час щоденної роботи всі 42 сигнали оновлюються наживо, а
          <code>ProductTower</code> кодує будь-який новий товар через його опис і 65 специфікацій навіть без перенавчання
          (у фінальному тесті: <strong>83.03% Recall@10</strong> на навчених клієнтах і товарах та <strong>76.63%</strong>
          на всіх 13 184 замовленнях повного каталогу).
        </div>
      </section>

      <!-- 7. TWO DEPLOYMENT SOLUTIONS -->
      <section class="section" id="deployment">
        <h2>7. Два варіанти впровадження системи в роботу</h2>
        <p>
          На фінальному тесті ми чітко побачили різницю між двома сценаріями: коли клієнт і товар уже були в навчанні
          (<strong>83.03% Recall@10</strong> на тесті vs <strong>84.27%</strong> на валідації) і коли трапляється абсолютно новий
          клієнт або новий товар після дати навчання (<strong>~57.8% Recall@10</strong>). Тому для реального впровадження в CRM / 1C
          пропонуються <strong>два практичні рішення</strong>:
        </p>
        <div class="arch-grid">
          <div class="block-card" style="background:var(--green-bg);border-color:var(--green-border);">
            <div class="block-num" style="background:#ffffff;color:var(--green);">Варіант 1 · Режим високої точності</div>
            <h3>Працювати лише для навчених клієнтів і товарів</h3>
            <p style="margin-bottom:8px;">
              Рекомендації автоматично показуються лише для клієнтів та позицій, що вже були присутні в останній навчальній вибірці
              (<code>&lt; T_cutoff</code>). Нові клієнти та щойно додані товари підключаються після чергового планового перенавчання
              (наприклад, раз на місяць або квартал).
            </p>
            <ul style="font-size:13.5px;margin-bottom:0;">
              <li><strong>Очікувана точність:</strong> <strong>83.0%–84.3% Recall@10</strong> (максимальна стабільність).</li>
              <li><strong>Перевага:</strong> висока довіра менеджерів, мінімум нерелевантних підказок.</li>
              <li><strong>Обмеження:</strong> не покриває найновіших клієнтів і товари до наступного оновлення моделі.</li>
            </ul>
          </div>
          <div class="block-card" style="background:var(--amber-bg);border-color:var(--amber-border);">
            <div class="block-num" style="background:#ffffff;color:var(--amber);">Варіант 2 · Універсальний режим + Попередження ⚠️</div>
            <h3>Працювати для всіх, але позначати нові сутності знаком ⚠️</h3>
            <p style="margin-bottom:8px;">
              Система видає прогнози для <strong>100% клієнтів і товарів</strong>: нові продукти кодуються через <code>ProductTower</code>
              за їхнім описом і 65 характеристиками, а нові клієнти — за профілем, галуззю та регіоном. Якщо клієнт або товар ще не
              проходили навчання, інтерфейс чітко показує менеджеру попереджувальний знак:
            </p>
            <div style="background:#ffffff;border:1.5px solid #f59e0b;border-radius:9px;padding:9px 12px;margin:8px 0;font-size:13px;color:#92400e;font-weight:700;">
              ⚠️ Увага: новий клієнт або новий товар (поза навчальною вибіркою) — прогноз орієнтовний, рекомендовано перевірити вручну.
            </div>
            <ul style="font-size:13.5px;margin-bottom:0;">
              <li><strong>Очікувана точність:</strong> <strong>76.6%–82.3%</strong> загалом (<strong>~58%</strong> на нових сутностях).</li>
              <li><strong>Перевага:</strong> 100% охоплення бізнесу з першого дня + прозора підказка менеджеру, де варто бути уважнішим.</li>
              <li><strong>Рекомендація (дешеве оновлення):</strong> перенавчати модель <strong>раз на 2 місяці або щоразу після додавання ~10 нових товарів</strong> — це займає лічені хвилини, мінімізує частку кейсів із <code>⚠️</code> і постійно тримає якість біля рівня навчених клієнтів і товарів (<strong>~84–85% Recall@10</strong>, як <code>84.27%</code> на валідації).</li>
            </ul>
          </div>
        </div>
      </section>

      <!-- 8. HOW TO READ THE DEMO -->
      <section class="section" id="demo-guide">
        <h2>8. Як користуватися демо та розуміти показники</h2>
        <p>Відкрийте файл <a href="OPEN_DEMO.html" style="color:var(--primary);font-weight:700;">OPEN_DEMO.html</a> — на екрані показано реальні замовлення клієнтів із фінального тесту (липень–листопад 2026):</p>
        <ul>
          <li><span class="pill hit">✓ Вгадано · Повтор</span> — зелений рядок означає, що клієнт справді купив цей знайомий товар того дня, і модель поставила його в Топ-10.</li>
          <li><span class="pill newhit">★ Вгадано · Новий!</span> — жовтий рядок означає найцінніше влучання: клієнт купив цей товар <strong>вперше в житті</strong>, а нейромережа передбачила це заздалегідь.</li>
          <li><strong>Теги під назвою товару</strong> (наприклад, <code>Купував 4× · 32д тому</code>, <code>Схожі клієнти</code>, <code>Регіон 90д</code>, <code>Галузь</code>) пояснюють головні причини, чому модель підняла продукт уверх.</li>
          <li><strong>Перемикач режиму:</strong> «Змішаний (Повтори + Новинки)» показує повний робочий список, а «Лише нові товари (Крос-продажі)» приховує вже куплені позиції й показує чисті пропозиції для розширення асортименту клієнта.</li>
          <li><strong>Recall@10</strong> — який відсоток реально куплених клієнтом товарів потрапив у Топ-10 рекомендацій.</li>
        </ul>
      </section>

      <!-- 9. PASSPORT -->
      <section class="section" id="passport">
        <h2>9. Технічний паспорт фінальної моделі в пакеті</h2>
        <table>
          <tbody>
            <tr><td><strong>Ідентифікатор релізу</strong></td><td class="mono">__REF_ID__</td></tr>
            <tr><td><strong>Архітектура</strong></td><td class="mono">Wave2BE2ERanker (2× Memory Transformer + Unseen Products Head)</td></tr>
            <tr><td><strong>Кількість параметрів</strong></td><td class="mono">__TRAIN_PARAMS__ (навчання) / __REQ_PARAMS__ (на 1 запит)</td></tr>
            <tr><td><strong>Каталог продуктів</strong></td><td class="mono">1 141 канонічний продукт (1 660 SKU)</td></tr>
            <tr><td><strong>Валідація (навчені C&amp;P / повна)</strong></td><td class="mono">84.27% (15 343 зам., NDCG 67.18%) · Повна: 82.26% (16 623 зам., PureU 51.78%)</td></tr>
            <tr><td><strong>Фінальний тест (навчені C&amp;P / повна)</strong></td><td class="mono">83.03% (9 839 зам., NDCG 63.49%) · Повна: 76.63% (13 184 зам.)</td></tr>
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
      id: 'cust_feat', label: '1. Ознаки клієнта (CUST_7)',
      x: -8.4, y: 1.95, z: 0.1, dx: 1.75, dy: 1.5, dz: 1.05,
      top: '#7dd3fc', front: '#0ea5e9', side: '#0369a1',
      summary: '<strong>1. Вхідні дані та ознаки клієнта (CUST_7 + профіль):</strong> окремий блок сирих даних про компанію на дату замовлення — опис діяльності, географія, галузь і 7 динамічних фінансово-поведінкових метрик.',
      features: [
        'Текстовий опис компанії — BAAI/bge-m3 (1024D)',
        'Категоріальні ID: city_id (місто), region_id (регіон), industry_id (галузь)',
        'CUST_7 [0]: customer_age — тривалість співпраці з клієнтом',
        'CUST_7 [1]: customer_30d_activity — активність замовлень за 30 днів',
        'CUST_7 [2]: customer_90d_activity — активність замовлень за 90 днів',
        'CUST_7 [3]: cust_prepaid_or_cash_ratio — частка передплат / готівки',
        'CUST_7 [4]: cust_avg_deferral_days_norm — середня відстрочка платежу',
        'CUST_7 [5]: cust_last_deferral_days_norm — остання відстрочка платежу',
        'CUST_7 [6]: cust_overdue_unpaid_ratio — частка прострочених оплат',
      ],
      dims: 'Вхід: текст 1024D + 3 категорії + CUST_7 (7D) → передається в CustomerTower; CUST_7 також іде у білінійний bias Cross-Attention.',
      training: 'Усі 7 ознак CUST_7 розраховуються строго до дати замовлення (< as_of_date). Неінформативні 4 колонки з CUST_11 видалено на етапі абляції.',
    },
    {
      id: 'cust_tower', label: '2. CustomerTower',
      x: -5.65, y: 1.95, z: 0.1, dx: 1.7, dy: 1.5, dz: 1.35,
      top: '#60a5fa', front: '#2563eb', side: '#1e3a8a',
      summary: '<strong>2. Вежа клієнта (CustomerTower / Token-0 Encoder):</strong> перетворює текстовий опис, регіон, галузь та 7 фінансово-поведінкових ознак у єдиний 128-вимірний вектор Токена 0 напряму (без окремого Stage 1 residual MLP).',
      features: [
        'Текстова гілка (cust_text_proj): Linear 1024 → 128 + LayerNorm + GELU + Dropout(0.12)',
        'Таблична гілка (cust_tab_proj): concat[city(16D), region(16D), industry(16D), CUST_7(7D)] = 55D → 128D → 128D (CUST_7 входить сюди напряму!)',
        'Злиття (token0_fusion): concat[z_text(128D), z_tab(128D)] = 256D → Linear(128) + LayerNorm + GELU',
      ],
      dims: 'Вхід: [1024D текст, 3 ID (48D), 7D CUST_7] → Вихід: 128D → + type_emb[0] = Токен 0 (CUST_7 також передається у Cross-Attention key_statistics).',
      training: 'Навчається наскрізно (End-to-End) разом з усією моделлю; dropout = 0.12, weight_decay = 1e-2.',
    },
    {
      id: 'tok0', label: '3. Токен 0 (Профіль)',
      x: -3.25, y: 1.95, z: 0.1, dx: 1.45, dy: 1.5, dz: 1.35,
      top: '#93c5fd', front: '#3b82f6', side: '#1d4ed8',
      summary: '<strong>3. Токен 0 (Профіль клієнта):</strong> перша позиція у 33-токенній пам’яті клієнта. Ніколи не маскується — саме цей токен дозволяє моделі прогнозувати навіть для нових клієнтів із 0 покупок.',
      features: [
        'Вектор 128D від CustomerTower',
        '+ type_emb[0] (маркер типу «профіль клієнта»)',
        'Ключ уваги несе 7 ознак CUST_7 у 12D просторі пам’яті',
      ],
      dims: 'Форма: (B, 1, 128) → об’єднується з Токенами 1..32 у послідовність (B, 33, 128) для Transformer Шар 1.',
      training: 'Постійно активний (attention_mask = 1) під час навчання й інференсу.',
    },
    {
      id: 'prod_tower', label: '4. ProductTower (1 141×128D)',
      x: -8.4, y: -2.25, z: 0.1, dx: 1.9, dy: 1.65, dz: 1.45, stacks: 4,
      top: '#fbbf24', front: '#f59e0b', side: '#b45309',
      summary: '<strong>4. Вежа продукту (ProductTower · StaticEnrichedProductTower):</strong> будує спільний 128D вектор-паспорт v_bn для кожного з 1 141 товарів. Подається ОДНОЧАСНО в Історію покупок (Токени 1..32) та у Cross-Attention як запити Q = v_bn!',
      features: [
        'Назва продукту без фасування — BAAI/bge-m3 (1024D)',
        '65D структурованих специфікацій: container_type (12D), volume tiers (8D), application_domain (11D), viscosity_grade (34D)',
        '6 категоріальних ембедінгів (16D кожен): product_category, product_subcategory, business_line, tier, unit, country_of_origin',
        'НУЛЬ ID-ембедінгів окремих товарів → здатен кодувати абсолютно нові товари за описом і специфікаціями без перенавчання!',
      ],
      dims: 'Вихід: v_bn форми (1 141, 128). Кешується офлайн (242 тис. параметрів вежі не запускаються на кожен запит → 2.9 мс CPU).',
      training: 'Навчається наскрізно (End-to-End) разом із ранжирувальником; дропаут = 0.12.',
    },
    {
      id: 'hist', label: '5. Історія (Токени 1..32 + SEQ_5)',
      x: -5.55, y: -0.45, z: 0.1, dx: 1.8, dy: 1.6, dz: 1.45, stacks: 4,
      top: '#34d399', front: '#10b981', side: '#047857',
      summary: '<strong>5. Історія покупок (Токени 1..32 + SEQ_5):</strong> останні 32 замовлені клієнтом продукти. Кожен токен бере спільний вектор v_bn напряму з ProductTower + позиційне кодування + 5 часових/об’ємних ознак SEQ_5.',
      features: [
        'Спільний вектор товару v_bn[seq_product_id] (128D) напряму з ProductTower',
        '+ pos_emb (позиція 1..32) + type_emb[1] (маркер «історія покупок»)',
        'SEQ_5 [0]: qty_log1p_norm — нормована кількість у замовленні',
        'SEQ_5 [1]: live_age — скільки днів минуло від цієї покупки до дати прогнозу',
        'SEQ_5 [5]: prior_bn_count_norm — накопичена к-сть покупок цього товару',
        'SEQ_5 [6]: prior_bn_vol_share — частка товару в об’ємі клієнта',
        'SEQ_5 [8]: real_event_indicator — індикатор реальної події (не паддінг)',
      ],
      dims: 'Форма: (B, 32, 128) + маска валідних подій + (B, 32, 5) SEQ_5 для білінійного bias Cross-Attention.',
      training: 'Календарні ознаки month_sin/cos і gap_days видалено після абляції; усі події строго < дати замовлення.',
    },
    {
      id: 'mem1', label: '6А. Transformer · Шар 1',
      x: -2.65, y: -0.45, z: 0.1, dx: 1.2, dy: 2.35, dz: 1.95,
      top: '#a5b4fc', front: '#6366f1', side: '#4338ca',
      summary: '<strong>6А. Пам’ять клієнта · Transformer Шар 1:</strong> приймає об’єднану стрічку з 33 токенів (Токен 0 профілю + 32 токени історії) та виявляє зв’язки між минулими покупками.',
      features: [
        'Вхід: 33 токени (Токен 0 від CustomerTower + Токени 1..32 з ProductTower)',
        '8-голова Self-Attention по всіх 33 токенах із маскою паддінгу',
        'Feed-Forward мережа: 128 → 384 → 128 (GELU + LayerNorm + Residual)',
      ],
      dims: 'Форма: (B, 33, 128) → (B, 33, 128), hidden FFN = 384.',
      training: 'Dropout = 0.12 на увазі та FFN; Pre-LN стабілізує градієнти (grad_clip = 1.0).',
    },
    {
      id: 'mem2', label: '6Б. Transformer · Шар 2',
      x: -1.05, y: -0.45, z: 0.1, dx: 1.2, dy: 2.35, dz: 1.95,
      top: '#818cf8', front: '#4f46e5', side: '#312e81',
      summary: '<strong>6Б. Пам’ять клієнта · Transformer Шар 2:</strong> другий глибокий блок пам’яті узагальнює довгі залежності, цикли поповнення та зміну брендів перед перехресною увагою.',
      features: [
        'Вхід: збагачені 33 токени з Шару 1',
        '8-голова Self-Attention + FFN (128 → 384 → 128)',
        'Вихідні ключі (K) та значення (V) для Cross-Attention',
      ],
      dims: 'Форма: (B, 33, 128). Додавання 2-го шару пам’яті підняло і Recall@10, і New@10 без перенавчання.',
      training: 'Dropout = 0.12; навчання 10 епох на 137 765 замовленнях із CosineAnnealingLR (eta_min = 1e-5).',
    },
    {
      id: 'sig42', label: '7. 42 стат. ознаки (cand_42)',
      x: 1.5, y: -2.45, z: 0.1, dx: 2.3, dy: 1.45, dz: 1.15,
      top: '#38bdf8', front: '#0284c7', side: '#075985',
      summary: '<strong>7. 42 статистичні каузальні ознаки (cand_42 · в парі з Cross-Attention):</strong> підключені безпосередньо до 8-голової перехресної уваги (як білінійний bias) та фінальних голів оцінки. Не входять у ProductTower чи CustomerTower!',
      features: [
        'Повтори (5): has_bought, order_count, qty_share, days_since_last, urgency',
        'Схожість із клієнтом (8): cat/subcat/brand/family shares, is_repeat_cat/brand/line, family_age & overdue',
        'Глобальний попит (7): global_pop, orders_7d/30d/90d, buyers_30d, cat_rel_share, global_sale_age',
        'Послідовності (7): last_lift, recent3_lift, cat/family/line_lift, support_confidence, same_basket_lift',
        'Менеджер, регіон і галузь (9): mgr_90d/30d, mgr_brand/cat_90d, mgr×ind_90d, region_90d, industry_affinity, viscosity, equipment_match',
        'Колаборативна фільтрація (2): full_cf_cosine, full_cf_recency_weighted (HL=180д) + category_age/overdue (4)',
      ],
      dims: 'Форма: (B, 1 141, 42) float16 → Білінійна матриця bias W_h: 8 × (42+1) × (5+7) прямо в Cross-Attention.',
      training: 'Усі 42 ознаки строго каузальні (< дати замовлення). 2 слабкі ознаки росту (growth_30d) видалено абляцією.',
    },
    {
      id: 'xatt', label: '8. 8-голова Cross-Attention',
      x: 1.5, y: -0.5, z: 0.1, dx: 2.3, dy: 3.05, dz: 2.25, stacks: 8,
      top: '#60a5fa', front: '#2563eb', side: '#1e3a8a',
      summary: '<strong>8. 8-голова перехресна увага (Cross-Attention + Full-42 Bias):</strong> «серце» моделі. Кожен із 1 141 товарів (Чистий запит Q = v_bn із ProductTower) звертається до 33 токенів пам’яті, а 42 статистичні ознаки керують вагою кожної з 8 голів.',
      features: [
        'Запити Q = v_bn (1 141 × 128D) напряму з ProductTower (без зайвого candidate_mlp)',
        'Ключі K і Значення V (33 × 128D) із 2-шарового Transformer пам’яті',
        'Full-42 Bilinear Bias: B_h(c, m) = [1, cand_42] · W_h · [SEQ_5 | CUST_7] окремо для кожної з 8 голів',
      ],
      dims: 'Вхід: Q(1 141×128), K/V(33×128), Bias(1 141×43 × 33×12) → Вихід: контекст h_att (1 141 × 128D).',
      training: 'Dropout = 0.12; кожен кандидат оцінюється паралельно без обрізання пулу (100% каталогу).',
    },
    {
      id: 'main', label: '9А. Main Head (Повтори)',
      x: 4.95, y: 1.1, z: 0.1, dx: 1.65, dy: 1.75, dz: 1.5,
      top: '#34d399', front: '#10b981', side: '#047857',
      summary: '<strong>9А. Голова 1 · Main Head (Регулярні повтори):</strong> глибока MLP-мережа для регулярного поповнення запасів, навчена з акцентом на свіжі продажі.',
      features: [
        'Конкатенація [v_bn (128), h_att (128), v_bn ⊙ h_att (128), cand_42 (42)] = 426D',
        'MLP: 426 → 384 → 192 → 96 → 1 (логіт повтору s_main)',
        'Допоміжна 60-денна голова під час навчання (aux_60d, λ = 0.3, вимкнена на інференсі)',
      ],
      dims: 'Вхід: 426D × 1 141 → Вихід: s_main (1 141 скалярних оцінок).',
      training: 'Dropout = 0.12; WeightedRankerLoss (BCE pos_weight=10 + Listwise Softmax CE) із експоненційною вагою свіжості HL=365д (2.31× на 2026-01 → 0.28× на 2023-01).',
    },
    {
      id: 'opta', label: '9Б. Unseen Products Head',
      x: 4.95, y: -1.45, z: 0.1, dx: 1.65, dy: 1.75, dz: 1.5,
      top: '#fcd34d', front: '#f59e0b', side: '#b45309',
      summary: '<strong>9Б. Голова 2 · Unseen Products Head (Нові товари / Cross-Sell):</strong> окрема голова, яка отримує лише 37 неповторних ознак і навчається виключно на товарах, які клієнт купує ВПЕРШЕ.',
      features: [
        'Конкатенація [v_bn (128), h_att (128), v_bn ⊙ h_att (128), cand_non_repeat_37 (37)] = 421D',
        '5 повторних колонок (has_bought, order_count, qty_share, days_since, urgency) повністю виключені!',
        'Окрема функція втрат (separated conflict-free targets) з вагою w_optA = 1.0',
      ],
      dims: 'Вхід: 421D × 1 141 → Вихід: s_unseen (оцінка відкриття нового товару).',
      training: '0% tabular dropout (unseen_dropout_cold_prob = 0.0) — дало максимальну стабільність крос-продажів; вага свіжості HL=365д.',
    },
    {
      id: 'out', label: '10. Калібрування + SKU',
      x: 7.45, y: -0.3, z: 0.15, dx: 1.4, dy: 2.3, dz: 2.05,
      top: '#3b82f6', front: '#1d4ed8', side: '#1e3a8a',
      summary: '<strong>10. Калібрування + Підбір фасування (SKU Resolver):</strong> зводить оцінки двох голів у єдиний Топ-10 та перетворює кожен продукт на конкретну тару клієнта.',
      features: [
        'Для вже куплених товарів: s_final = s_main',
        'Для нових товарів: s_final = a · s_unseen + b (заморожені параметри a = 0.940, b = −2.009)',
        'Каузальний Pack-Size SKU Resolver: історія тари клієнта → улюблений тип тари → популярність SKU',
      ],
      dims: 'Вихід: Топ-10 унікальних продуктів + 10 точних артикулів (SKU) за ~2.9 мс на CPU.',
      training: 'Параметри (a, b) підбирались 2-фолдовою customer-grouped OOF-калібровкою виключно на валідації (нуль дотику до тесту).',
    },
  ];

  const EDGES = [
    ['cust_feat', 'cust_tower', '#0ea5e9'],
    ['cust_tower', 'tok0', '#2563eb'],
    ['tok0', 'mem1', '#3b82f6'],
    ['prod_tower', 'hist', '#f59e0b'],
    ['prod_tower', 'xatt', '#d97706'],
    ['hist', 'mem1', '#10b981'],
    ['mem1', 'mem2', '#4f46e5'],
    ['mem2', 'xatt', '#4f46e5'],
    ['sig42', 'xatt', '#0284c7'],
    ['sig42', 'opta', '#0284c7'],
    ['xatt', 'main', '#10b981'],
    ['xatt', 'opta', '#f59e0b'],
    ['main', 'out', '#10b981'],
    ['opta', 'out', '#f59e0b'],
  ];

  const byId = Object.fromEntries(BLOCKS.map(b => [b.id, b]));
  // Camera angles: yaw orbits horizontally around Z, elev tilts down from above (0.10..1.38 rad)
  let yaw = -0.32, elev = 0.48, zoom = 0.96, autoSpin = false, selectedId = 'xatt';
  let dragging = false, lx = 0, ly = 0, draggedDist = 0;
  let hitZones = [];
  let tAnim = 0;

  function renderInspector(b) {
    if (!infoEl || !b) return;
    const isCompact = canvasId.startsWith('pres-');
    if (isCompact) {
      infoEl.innerHTML = `${b.summary} <span style="color:#475569;">(${b.dims})</span>`;
      return;
    }
    const featLis = b.features.map(f => `<li style="margin-bottom:3px;">${f}</li>`).join('');
    infoEl.innerHTML = `
      <div style="margin-bottom:8px;">${b.summary}</div>
      <details style="margin-top:6px;background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:7px 11px;">
        <summary style="cursor:pointer;font-weight:700;color:#1e3a8a;font-size:13px;">📋 Вхідні ознаки та компоненти блоку (${b.features.length})</summary>
        <ul style="margin:7px 0 2px 18px;font-size:12.8px;color:#334155;">${featLis}</ul>
      </details>
      <details style="margin-top:6px;background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:7px 11px;">
        <summary style="cursor:pointer;font-weight:700;color:#1e3a8a;font-size:13px;">📐 Розмірності тензорів і зв'язки</summary>
        <div style="margin-top:6px;font-size:12.8px;color:#334155;font-family:var(--mono);">${b.dims}</div>
      </details>
      <details style="margin-top:6px;background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:7px 11px;">
        <summary style="cursor:pointer;font-weight:700;color:#1e3a8a;font-size:13px;">⚙️ Навчання, Dropout та регуляризація</summary>
        <div style="margin-top:6px;font-size:12.8px;color:#334155;">${b.training}</div>
      </details>
    `;
  }

  renderInspector(byId[selectedId]);
  if (autoBtn) autoBtn.textContent = 'Авто-обертання: Вимк.';

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

  // True right-handed camera projection:
  //   - Rotate world (x, y, z) around vertical Z axis by yaw
  //   - Tilt camera down by elevation angle `elev`
  //   - +depth = FARTHER from camera, -depth = CLOSER to camera
  //   - Perspective scale = camDist / (camDist + depth) so closer objects appear larger
  function proj(x, y, z, W, H) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const ce = Math.cos(elev), se = Math.sin(elev);
    const x1 = x * cy - y * sy;
    const y1 = x * sy + y * cy;
    const depth = y1 * ce - z * se;
    const up = y1 * se + z * ce;
    const camDist = 26.0;
    const persp = camDist / Math.max(4.0, camDist + depth);
    const scale = Math.min(W / 20.4, H / 8.4) * zoom * persp;
    return { sx: W * 0.5 + x1 * scale, sy: H * 0.57 - up * scale, depth, scale };
  }

  function centerOf(b) {
    return [b.x + b.dx * 0.5, b.y + b.dy * 0.5, b.z + b.dz * 0.5];
  }

  // Face normals in world space + CCW vertex order viewed from OUTSIDE the box
  const BOX_FACES = [
    { idx: [4, 5, 6, 7], n: [0, 0, 1], kind: 'top' },    // +Z top
    { idx: [0, 3, 2, 1], n: [0, 0, -1], kind: 'side' },  // -Z bottom
    { idx: [0, 1, 5, 4], n: [0, -1, 0], kind: 'front' }, // -Y front
    { idx: [2, 3, 7, 6], n: [0, 1, 0], kind: 'front' },  // +Y back
    { idx: [1, 2, 6, 5], n: [1, 0, 0], kind: 'side' },   // +X right
    { idx: [3, 0, 4, 7], n: [-1, 0, 0], kind: 'side' },  // -X left
  ];

  function drawPrism(x, y, z, dx, dy, dz, topC, frontC, sideC, isSel, W, H) {
    const pts = [
      [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
      [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ].map(p => proj(p[0], p[1], p[2], W, H));

    // Only draw front-facing faces (true 2D signed screen area check = exact back-face culling)
    const visible = [];
    for (const f of BOX_FACES) {
      const p0 = pts[f.idx[0]], p1 = pts[f.idx[1]], p2 = pts[f.idx[2]];
      const cross = (p1.sx - p0.sx) * (p2.sy - p0.sy) - (p1.sy - p0.sy) * (p2.sx - p0.sx);
      // In screen coords (Y down), CCW outside-facing vertices have negative cross product
      if (cross < 0) {
        const avgDepth = f.idx.reduce((s, i) => s + pts[i].depth, 0) / 4;
        const col = f.kind === 'top' ? topC : (f.kind === 'front' ? frontC : sideC);
        visible.push({ idx: f.idx, col, avgDepth });
      }
    }
    // Painter's order: draw FARTHER faces first (larger depth -> smaller depth)
    visible.sort((a, b) => b.avgDepth - a.avgDepth);

    for (const f of visible) {
      ctx.beginPath();
      f.idx.forEach((vi, k) => k === 0 ? ctx.moveTo(pts[vi].sx, pts[vi].sy) : ctx.lineTo(pts[vi].sx, pts[vi].sy));
      ctx.closePath();
      ctx.fillStyle = f.col;
      ctx.globalAlpha = isSel ? 1.0 : 0.94;
      ctx.fill();
      ctx.strokeStyle = isSel ? '#0f172a' : 'rgba(255,255,255,0.85)';
      ctx.lineWidth = isSel ? 1.7 : 0.9;
      ctx.stroke();
    }
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

      if (autoSpin && !dragging) yaw = -0.34 + Math.sin(tAnim * 0.42) * 0.22;
      tAnim += 0.016;

      // Subtle 3D floor grid at z = 0
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 0.85;
      for (let gx = -8; gx <= 9; gx += 1.5) {
        const a = proj(gx, -3.2, 0, W, H), b = proj(gx, 3.8, 0, W, H);
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
      }
      for (let gy = -3; gy <= 3.6; gy += 1.1) {
        const a = proj(-8, gy, 0, W, H), b = proj(9, gy, 0, W, H);
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
      }

      // Soft ground shadows under each block
      BLOCKS.forEach(b => {
        const sPts = [
          [b.x + 0.12, b.y - 0.12, 0.01], [b.x + b.dx + 0.12, b.y - 0.12, 0.01],
          [b.x + b.dx + 0.12, b.y + b.dy - 0.12, 0.01], [b.x + 0.12, b.y + b.dy - 0.12, 0.01],
        ].map(p => proj(p[0], p[1], p[2], W, H));
        ctx.beginPath();
        sPts.forEach((p, k) => k === 0 ? ctx.moveTo(p.sx, p.sy) : ctx.lineTo(p.sx, p.sy));
        ctx.closePath();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.07)';
        ctx.fill();
      });

      // Draw 3D flow lines & animated data pulses
      EDGES.forEach(([u, v, col], ei) => {
        const p1 = centerOf(byId[u]), p2 = centerOf(byId[v]);
        const s1 = proj(p1[0], p1[1], p1[2], W, H);
        const s2 = proj(p2[0], p2[1], p2[2], W, H);
        ctx.beginPath();
        ctx.moveTo(s1.sx, s1.sy);
        ctx.lineTo(s2.sx, s2.sy);
        ctx.strokeStyle = col;
        ctx.globalAlpha = 0.38;
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

      // True painter's order: draw FARTHER blocks first (larger depth -> smaller depth)
      const sorted = BLOCKS.map(b => {
        const c = centerOf(b);
        return { b, depth: proj(c[0], c[1], c[2], W, H).depth };
      }).sort((a, b) => b.depth - a.depth);

      hitZones = [];
      const labels = [];
      sorted.forEach(({ b }) => {
        const isSel = b.id === selectedId;
        let bbox;
        if (b.stacks && b.stacks > 1) {
          const gap = 0.055;
          const sh = (b.dz - gap * (b.stacks - 1)) / b.stacks;
          let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, topCx = 0, topCy = 0;
          for (let s = 0; s < b.stacks; s++) {
            const z0 = b.z + s * (sh + gap);
            const sb = drawPrism(b.x, b.y, z0, b.dx, b.dy, sh, b.top, b.front, b.side, isSel, W, H);
            minX = Math.min(minX, sb.minX); maxX = Math.max(maxX, sb.maxX);
            minY = Math.min(minY, sb.minY); maxY = Math.max(maxY, sb.maxY);
            topCx = sb.topCx; topCy = sb.topCy;
          }
          bbox = { minX, maxX, minY, maxY, topCx, topCy };
        } else {
          bbox = drawPrism(b.x, b.y, b.z, b.dx, b.dy, b.dz, b.top, b.front, b.side, isSel, W, H);
        }
        hitZones.push({ id: b.id, ...bbox });
        labels.push({ b, isSel, ax: bbox.topCx, ay: bbox.minY, x: bbox.topCx, y: bbox.minY - 14 });
      });

      // Repel overlapping label pills vertically so labels stay crisp and legible
      const placed = [];
      labels.sort((a, b) => a.y - b.y);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      labels.forEach(item => {
        ctx.font = item.isSel ? '700 11px -apple-system, sans-serif' : '600 10px -apple-system, sans-serif';
        const tw = ctx.measureText(item.b.label).width + 12;
        const th = 19;
        let lx0 = Math.max(6, Math.min(W - tw - 6, item.x - tw / 2));
        let ly0 = Math.max(6, Math.min(H - th - 6, item.y - th / 2));
        for (let iter = 0; iter < 6; iter++) {
          let hit = false;
          for (const p of placed) {
            if (lx0 < p.x + p.w + 4 && lx0 + tw + 4 > p.x && ly0 < p.y + p.h + 3 && ly0 + th + 3 > p.y) {
              ly0 = Math.max(6, p.y - th - 4);
              hit = true;
            }
          }
          if (!hit) break;
        }
        placed.push({ x: lx0, y: ly0, w: tw, h: th });

        // Leader hairline from pill to block top
        ctx.beginPath();
        ctx.moveTo(lx0 + tw / 2, ly0 + th);
        ctx.lineTo(item.ax, item.ay);
        ctx.strokeStyle = item.isSel ? '#0f172a' : '#94a3b8';
        ctx.lineWidth = 0.9;
        ctx.stroke();

        ctx.fillStyle = item.isSel ? '#0f172a' : 'rgba(255,255,255,0.95)';
        ctx.strokeStyle = item.isSel ? '#0f172a' : '#cbd5e1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(lx0, ly0, tw, th, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = item.isSel ? '#ffffff' : '#0f172a';
        ctx.fillText(item.b.label, lx0 + tw / 2, ly0 + th / 2 + 0.5);
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
    yaw -= dx * 0.008;
    elev = Math.max(0.10, Math.min(1.38, elev + dy * 0.008));
  });
  canvas.addEventListener('pointerup', e => {
    dragging = false;
    canvas.style.cursor = 'grab';
    if (draggedDist < 6) {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      // Hit test front-most block first (reverse of painter's order)
      for (let i = hitZones.length - 1; i >= 0; i--) {
        const z = hitZones[i];
        if (mx >= z.minX - 6 && mx <= z.maxX + 6 && my >= z.minY - 24 && my <= z.maxY + 6) {
          selectedId = z.id;
          renderInspector(byId[z.id]);
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
    yaw = -0.32; elev = 0.48; zoom = 0.96; autoSpin = false;
    if (autoBtn) autoBtn.textContent = 'Авто-обертання: Вимк.';
  });
  if (topBtn) topBtn.addEventListener('click', () => {
    yaw = 0.0; elev = 1.32; zoom = 1.02; autoSpin = false;
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
