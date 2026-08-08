# Micro Pullback Scanner & Predictor

סורק דפוסי **Micro Pullback** על נתונים יומיים + מודל שמעריך את ההסתברות
ש־pullback פעיל יתפתח לפריצה כלפי מעלה.

## הדפוס

1. **רגל אימפולס** — עלייה חזקה (‎5%+ ב־10 ימים, רוב הנרות ירוקים, מחיר מעל EMA20 ו־SMA50)
2. **Pullback רדוד** — 1–4 נרות אדומים/קטנים, ריטרייס עד 61.8% מהרגל, מחזיק מעל EMA20, במחזור מסחר נמוך מהאימפולס
3. **אישור** — סגירה מעל השיא של ה־pullback בתוך 5 ימים

## התקנה

```bash
pip install -r requirements.txt
```

## שימוש

```bash
# רשימת S&P 500 (נשמרת ל-sp500.txt)
python micro_pullback.py universe

# סריקת דפוסים היסטוריים + סטאפים פעילים
python micro_pullback.py scan --tickers PAYX,MSFT,NVDA,AAPL --period 2y

# backtest מלא ב-R עם עלויות (תוחלת, profit factor, drawdown, פירוט שנתי)
python micro_pullback.py backtest --tickers-file sp500.txt --period 10y

# הערכת walk-forward: האם המודל מוסיף ערך מחוץ למדגם?
python micro_pullback.py evaluate --tickers-file sp500.txt --period 10y --dataset trades.json

# אימון המודל הסופי (רגרסור R + סף EV, נשמרים יחד)
python micro_pullback.py train --dataset trades.json

# סטאפים פעילים עכשיו עם R חזוי והכרעת TAKE/skip
python micro_pullback.py predict --tickers-file sp500.txt --period 2y
```

הורדות נשמרות ב-cache דיסק (`cache/`, ניתן לשינוי עם `--cache-dir`) —
הרצה חוזרת לא מורידה שוב.

עבודה בלי אינטרנט — קבצי CSV בפורמט `Date,Open,High,Low,Close,Volume`:

```bash
python micro_pullback.py scan  --csv data/PAYX.csv
python micro_pullback.py train --csv data_dir/          # תיקייה שלמה
```

## קונפיגורציה של פרמטרי הדפוס

כל פרמטרי הזיהוי ניתנים לשינוי בשתי דרכים (CLI גובר על קובץ, קובץ גובר על ברירות מחדל):

```bash
# דגלים ב-CLI
python micro_pullback.py scan --tickers PAYX --impulse-min-gain 0.08 --max-retrace 0.5

# קובץ json (ראו mp_config.example.json)
python micro_pullback.py scan --tickers PAYX --config my_config.json
```

| פרמטר | ברירת מחדל | תיאור |
|---|---|---|
| `impulse_days` | 10 | חלון רגל האימפולס בימים |
| `impulse_min_gain` | 0.05 | עלייה מינימלית ברגל (5%) |
| `impulse_min_green` | 0.55 | שיעור מינימלי של נרות ירוקים ברגל |
| `impulse_min_rvol` | 1.2 | נפח הרגל ביחס לנפח הממוצע (20 יום) שלפניה — מסנן אימפולסים בנפח דליל (0 = כבוי) |
| `pullback_min_len` | 1 | אורך pullback מינימלי בנרות |
| `pullback_max_len` | 4 | אורך pullback מקסימלי בנרות |
| `max_retrace` | 0.618 | ריטרייס מקסימלי מטווח הרגל |
| `vol_contraction` | 1.10 | תקרת יחס מחזור pullback/אימפולס |
| `confirm_within` | 5 | ימים מקסימליים עד פריצה מאשרת |
| `label_horizon` | 10 | חלון תיוג למודל (ימים קדימה) |
| `label_target_atr` | 1.5 | יעד הצלחה ב-ATR מעל שיא ה-pullback |
| `label_stop_atr` | 1.0 | סטופ תיוג ב-ATR מתחת לנמוך ה-pullback |

`train` שומר את הפרמטרים בתוך קובץ המודל, ו-`predict` משתמש אוטומטית באותם
פרמטרים שאיתם המודל אומן — כדי שהזיהוי והחיזוי יהיו עקביים. אפשר לעקוף גם
שם עם דגלים מפורשים (תודפס אזהרה).

## המודל (meta-labeling)

הדפוס נותן את הסיגנל; המודל מחליט **לקחת או לדלג**. כל עסקה מדומה
(כניסה בפריצת הטריגר, סטופ 1R מתחת לנמוך הנסיגה, יעד 2R, יציאת זמן
אחרי 15 יום, עלויות 10bps) מקבלת תוצאה ב-R, ואנסמבל של רגרסורים
(HistGradientBoosting, 5 seeds) לומד לחזות את ה-R הצפוי מתוך הפיצ'רים —
כולם ידועים בזמן ההחלטה:

- **דפוס**: עומק ריטרייס, אורך נסיגה, יחס מחזורים, rvol של האימפולס,
  שיפוע נפח בנסיגה, מיקום הסגירה בנר האחרון, RSI, מרחק מ-EMA10/20, ATR
- **הקשר מניה**: מרחק משיא 52 שבועות, חוזק יחסי מול SPY (63 יום)
- **הקשר שוק**: מגמת SPY מול SMA200/SMA50, תשואת SPY 20 יום,
  רמת VIX ואחוזון VIX שנתי

סף הכניסה נבחר לפי מקסימום תוחלת (avg R) על סט האימון מתוך רשת
קוונטיילים, ונשמר יחד עם המודל.

## תוצאות walk-forward (S&P 500, 2016–2026)

הערכה כרונולוגית אמיתית: לכל שנת בדיקה המודל אומן רק על שנים קודמות
(עם purge בגבול), והסף נבחר על האימון בלבד. על 5,832 עסקאות:

| | בסיס (כל העסקאות) | מסונן (המודל) |
|---|---|---|
| תוחלת לעסקה | ‎+0.062R | **+0.156R** |
| שנים מנצחות | — | **9/9** |
| bootstrap p-value | — | **0.039** |

המודל הוכיח ערך עקבי מחוץ למדגם. עדיין: ביצועי עבר אינם ערובה לעתיד,
היקום סובל מהטיית שרידות (S&P 500 של היום), והתוחלת רגישה לעלויות בפועל.

לבדיקה מקומית ללא רשת יש מחולל נתונים סינתטיים:

```bash
python make_test_data.py 12
python micro_pullback.py train --csv test_data
```

> ⚠️ כלי מחקר בלבד — לא ייעוץ השקעות.
