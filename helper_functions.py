# -*- coding: utf-8 -*-
# IMPORT LIBRARIES
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import json
import altair as alt
import yfinance as yf

# FUNKTION, UM FILES ZU ÖFFNEN
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
# FUNKTION, UM JSON FILES LADEN
def load_ticker_list(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        ticker_dict = json.load(f)
    return ticker_dict

# FUNKTION, UM HISTORISCHE DATEN FÜR BESTIMMTE COMPANY ZU LADEN
def get_stock_price_range(ticker: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    """
    Historische Daten holen für ausgewählten Ticker.
    
    Args:
        ticker (str): Stock Ticker Symbol.
        start_date (str): Startdatum im Format 'YYYY-MM-DD' format.
        end_date (str): Enddatum im Format 'YYYY-MM-DD' format.
        
    Returns:
        Tuple: Mit den historischen Daten (als DataFrame) und dem Longname der Company.
    """
    try:
        # Create Ticker Object
        stock = yf.Ticker(ticker)

        # Fetch historical data
        hist = stock.history(start=start_date, end=end_date)

        # Fetch long name
        long_name = stock.info.get("longName", "N/A")

        if hist.empty:
            print("Keine Daten gefunden für diesen Zeitraum.")
            return None, None

        return hist, long_name

    except Exception as e:
        print(f"Error beim Abholen der Daten: {str(e)}")
        return None, None
    
# FUNKTION, UM DATEN ZU SPEICHERN (OPTIONAL)
def save_to_csv(data: pd.DataFrame, file_path: str, long_name: str) -> None:
    """
    Historische Daten als CSV-File speichern.
    
    Args:
        data (pd.DataFrame): Historische Daten.
        file_path (str): Pfad, wo CSV-File gespeichert wird
        long_name (str): Company lang Name als Spalte.
    """
    try:
        # Reset index damit Date Spalte wird
        data.reset_index(inplace=True)

        # Comapny Longname als Spalte
        data['Company'] = long_name

        # Reihenfolge Spalten
        formatted_data = data[['Date', 'Company', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Als CSV-File speichern
        formatted_data.to_csv(file_path, index=False)
        print(f"Daten erfolgreich gespeichert in {file_path}.")

    except Exception as e:
        print(f"Error beim Speichern der CSV: {str(e)}")
        
# REPORT ERZEUGEN
def generate_report(date, actual_price, predicted_price, days):
    print(colored("Erstelle Report...", "cyan"))
    
    error = abs(actual_price - predicted_price)
    error_percentage = (error / actual_price) * 100
    
    report = f"""
Aktiekurs Vorhersage Report
======================
Startdatum: {date.strftime('%Y-%m-%d')}
Vorhersagezeitraum: {days}
Zieldatum: {(date + pd.Timedelta(days=days)).strftime('%Y-%m-%d')}
Tatsöchlicher Preis: ${actual_price:.2f}
Vorhergesagter Preis: ${predicted_price:.2f}
Absoluter Error: ${error:.2f}
Prozentualer Error: {error_percentage:.2f}%
"""
    
    with open('REPORTS/prediction_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

# GEILER PLOT
def plot_predictions_with_actuals_and_deviation(stock_data, predictions):
    """Visualize historical data, predictions, and deviations with clear separation."""
    import pandas as pd

    # Letztes tatsächliches Datum und Schlusskurs
    last_actual_date = stock_data['Date'].max()
    last_actual_close = stock_data.iloc[-1]['Close']

    # Vorhersage-Datenframe erstellen
    prediction_dates = [last_actual_date + pd.Timedelta(days=i + 1) for i in range(len(predictions))]
    predictions_df = pd.DataFrame({
        "Date": prediction_dates,
        "Close": predictions
    })

    # Abweichungs-Datenframe erstellen
    deviations_df = pd.DataFrame({
        "Date": prediction_dates,
        "Actual": [last_actual_close] * len(predictions),
        "Predicted": predictions
    })

    # Historische Daten (blaue Linie)
    historical_chart = alt.Chart(stock_data).mark_line(color="blue").encode(
        x=alt.X('Date:T', axis=alt.Axis(title="Datum")),
        y=alt.Y('Close:Q', axis=alt.Axis(title="Schlusskurs")),
        tooltip=["Date:T", "Close"]
    )

    # Vorhersagedaten (rote Linie + Punkte)
    prediction_line = alt.Chart(predictions_df).mark_line(color="red").encode(
        x=alt.X('Date:T'),
        y=alt.Y('Close:Q'),
        tooltip=["Date:T", "Close"]
    )

    prediction_points = alt.Chart(predictions_df).mark_point(color="red", size=80).encode(
        x=alt.X('Date:T'),
        y=alt.Y('Close:Q'),
        tooltip=["Date:T", "Close"]
    )

    # Abweichungslinien (orange gestrichelt)
    deviation_lines = alt.Chart(deviations_df).mark_rule(color="orange", strokeDash=[4, 4]).encode(
        x=alt.X('Date:T'),
        y='Actual:Q',
        y2='Predicted:Q',
        tooltip=["Date:T", "Actual", "Predicted"]
    )

    # Combine all layers
    return (historical_chart + prediction_line + prediction_points + deviation_lines).properties(
        width=800,
        height=400,
        title="Historische Kurse, Vorhersagen und Abweichungen"
    )

# FUNKTION, UM VERGLEICH PLOT ZU ZEICHNEN
def create_comparison_plot(actual_price, predicted_price, date, days):
    print(colored("Erstelle Balkendiagramm...", "cyan"))
    
    plt.figure(figsize=(10, 6))
    prices = [round(actual_price, 2), round(predicted_price, 2)]
    labels = ['Actual', 'Predicted']
    colors = ['blue', 'red']
    
    plt.bar(labels, prices, color=colors)
    plt.title(f'Aktienkurs Vergleich für {date.strftime("%Y-%m-%d")} (+{days} days)')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    
    # LABLES MIT 2 DEZIMALSTELLEN
    for i, price in enumerate(prices):
        plt.text(i, price, f'${price:.2f}', ha='center', va='bottom')
    
    plt.savefig('PLOTS/price_comparison.png')
    plt.close()

# FUNKTION, FÜR Time Series Plot
def plot_predictions_over_time(results_df):
    """
    Erzeugt eine Zeitreihe, die tatsächliche vs. vorhergesagte Preise vergleicht.

    Args:
        results_df (DataFrame): Enthält die Spalten 'Datum', 'Prediction', und 'Actual'.
    """
    results_df['date'] = pd.to_datetime(results_df['Datum'], dayfirst=True)
    results_df['predicted'] = results_df['Prediction'].str.replace('$', '').str.replace(',', '').astype(float)
    results_df['actual'] = results_df['Actual'].str.replace('$', '').str.replace(',', '').astype(float)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(results_df['date'], results_df['actual'], label='Actual Kurs', marker='o', linestyle='-')
    ax.plot(results_df['date'], results_df['predicted'], label='Predicted Kurs', marker='x', linestyle='--')

    ax.set_title('Actual vs. Predicted Kurs')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Preis (USD)')
    ax.legend()
    plt.xticks(rotation=45)
    return fig
