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
