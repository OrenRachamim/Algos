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
# סריקת דפוסים היסטוריים + סטאפים פעילים
python micro_pullback.py scan --tickers PAYX,MSFT,NVDA,AAPL --period 2y

# אימון מודל החיזוי (מומלץ סל רחב והיסטוריה ארוכה)
python micro_pullback.py train --tickers PAYX,MSFT,NVDA,AAPL,GOOG,AMZN,META,TSLA,JPM,V --period 5y

# הסתברות לעלייה עבור pullback פעיל עכשיו
python micro_pullback.py predict --tickers PAYX --period 2y
```

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

## המודל

GradientBoosting על פיצ'רים של כל אירוע: עומק הריטרייס, אורך ה־pullback,
יחס המחזורים, RSI, מרחק מ־EMA10/20, שיפוע המגמה, ATR יחסי ועוד.
תווית הצלחה: המחיר עולה ‎+1.5 ATR מעל שיא ה־pullback לפני שהוא יורד
‎-1.0 ATR מתחת לנמוך שלו, בחלון של 10 ימים. הפיצול לאימון/בדיקה כרונולוגי
(ללא זליגת עתיד).

לבדיקה מקומית ללא רשת יש מחולל נתונים סינתטיים:

```bash
python make_test_data.py 12
python micro_pullback.py train --csv test_data
```

> ⚠️ כלי מחקר בלבד — לא ייעוץ השקעות.
