import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

# Load station data
df_stations = pd.read_excel('Station_name_8477.xlsx')
df_stations.columns = [col.replace("\n", " ").strip() for col in df_stations.columns]

# Helper: Get station info (prioritize NSG1/NSG2)
def get_station_info(city_name):
    pattern = r'\b' + re.escape(city_name) + r'\b'

    # Step 1: Match city name in Station Name or District
    matches = df_stations[
        df_stations['Station Name'].str.contains(pattern, case=False, na=False, regex=True) |
        df_stations['District'].str.contains(pattern, case=False, na=False, regex=True)
    ]

    if not matches.empty:
        return sort_main_stations(matches)

    # Step 2: Match exact district
    matching_districts = df_stations[df_stations['District'].str.lower() == city_name.lower()]
    if not matching_districts.empty:
        return sort_main_stations(matching_districts)

    return pd.DataFrame(columns=['Station Name', 'Station Code'])

# ✅ FIXED: Prioritize main stations (NSG1 and NSG2 first)
def sort_main_stations(df):
    priority = {'NSG1': 1, 'NSG2': 2, 'NSG3': 3, 'NSG4': 4, 'NSG5': 5, 'NSG6': 6}

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Assign priority safely using .loc
    df.loc[:, '__priority__'] = df['New Station Category'].map(priority).fillna(999)

    sorted_df = df.sort_values(by='__priority__').drop(columns='__priority__')
    return sorted_df[['Station Name', 'Station Code']].drop_duplicates().reset_index(drop=True)

# Helper: Create station slug for URL
def format_station_slug(station_name, station_code):
    return f"{station_name.strip().replace('.', '').replace(' ', '-').upper()}-{station_code}"

# 🎯 Main Function
def get_etrain_trains(source_city:str, dest_city:str, user_date:str) -> pd.DataFrame:
    try:
        date_obj = datetime.strptime(user_date, "%d/%m/%Y")
    except ValueError:
        print("❌ Invalid date format. Use dd/mm/yyyy.")
        return pd.DataFrame()

    day = date_obj.day
    month = date_obj.strftime("%b")
    year = date_obj.year

    # Get station info
    src = get_station_info(source_city)
    dst = get_station_info(dest_city)

    if src.empty or dst.empty:
        print("❌ Source or destination station not found.")
        return pd.DataFrame()

    src_slug = format_station_slug(src.iloc[0]['Station Name'], src.iloc[0]['Station Code'])
    dst_slug = format_station_slug(dst.iloc[0]['Station Name'], dst.iloc[0]['Station Code'])
    url = f"https://etrain.info/trains/{src_slug}-to-{dst_slug}"
    print(f"[INFO] URL: {url}")

    # Launch browser
    driver = webdriver.Chrome()
    driver.get(url)
    wait = WebDriverWait(driver, 20)
    time.sleep(2)

    # Select calendar date
    date_icon = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".datepicker")))
    date_icon.click()
    time.sleep(1)

    # Loop to match month/year
    for _ in range(12):
        displayed_month_year = driver.find_element(By.CLASS_NAME, "monthDsp").get_attribute("value").strip()
        if displayed_month_year == f"{month} {year}":
            break
        next_btn = driver.find_element(By.CSS_SELECTOR, "input.nav[type='button'][value='>']")
        next_btn.click()
        time.sleep(0.5)

    # Select day
    day_cells = driver.find_elements(By.XPATH, f"//input[@type='button' and @value='{day}']")
    if not day_cells:
        print("❌ Unable to select the specified date.")
        driver.quit()
        return pd.DataFrame()
    day_cells[0].click()

    # Wait for train list
    time.sleep(2)
    rows = driver.find_elements(By.XPATH, '//div[@class="trnlstcont borderbottom rnd5 bx1s"]/div/table/tbody/tr')
    trains = []

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) >= 6:
            trains.append({
                "Train No": cols[0].text.strip(),
                "Train Name": cols[1].text.strip(),
                "From": cols[2].text.strip(),
                "Departs": cols[3].text.strip(),
                "To": cols[4].text.strip(),
                "Arrives": cols[5].text.strip()
            })

    return pd.DataFrame(trains)