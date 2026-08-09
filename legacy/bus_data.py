from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from datetime import datetime, timedelta
from langchain.tools import Tool
from time import sleep
import pandas as pd

def get_abhibus_buses(origin :str, destination :str, date_str :str)->dict:
    """
    Scrapes Abhibus.com for available buses between `origin` and `destination` on a specific `date_str` (DD/MM/YYYY).
    
    Returns:
        dict: Contains list of bus info or error message.
    """
    
    # -------------------- Setup WebDriver --------------------
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 15)
    driver.get("https://www.abhibus.com")

    def enter_location(xpath, location_name):
        input_elem = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        input_elem.clear()
        input_elem.click()
        input_elem.send_keys(location_name)
        sleep(1)
        input_elem.send_keys(Keys.DOWN)
        input_elem.send_keys(Keys.ENTER)

    try:
        # Step 1: Fill From & To
        print("🛣️ Entering locations...")
        from_xpath = '//*[@id="search-from"]/div[1]/div/div/div/div[2]/input'
        to_xpath = '//*[@id="search-to"]/div/div/div/div/div[2]/input'

        enter_location(from_xpath, origin)
        enter_location(to_xpath, destination)

        # Step 2: Set Date via JS
        date_input = driver.find_element(By.XPATH, '//input[@placeholder="Onward Journey Date"]')
        driver.execute_script("arguments[0].value = arguments[1];", date_input, date_str)
        driver.execute_script("""
            const evt = new Event('input', { bubbles: true });
            arguments[0].dispatchEvent(evt);
        """, date_input)
        print(f"📅 Date set to {date_str}")

        # Step 3: Click Search
        search_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="search-button"]/a')))
        search_btn.click()
        print("🔍 Triggering search...")
        sleep(5)

        # Get current URL and convert date format
        search_url = driver.current_url

        def update_abhibus_url(url, desired_date):
            return url.replace(url.split("/")[-2], desired_date.replace("/", "-"))

        updated_url = update_abhibus_url(search_url, date_str)
        driver.get(updated_url)
        print(f"📎 Updated booking URL: {updated_url}")
        sleep(3)

        # Scrape all required fields
        print("🔄 Collecting data...")
        company_elements = driver.find_elements(By.XPATH, '//h5[@class="title"]')
        companies = [elem.text for elem in company_elements]

        times = driver.find_elements(By.XPATH, '//div[@class="text-sm col auto"]/span')
        times = [time.text for time in times]
        dept_time = times[::2]
        arr_time = times[1::2]

        bus_fare = driver.find_elements(By.XPATH, '//span[@class="fare text-neutral-800"]')
        fare = [f.text for f in bus_fare]

        avl_seats = driver.find_elements(By.XPATH, '//div[@class="row seat-info bd-success-400 text-success-600 bg-success-50"]/div/div')
        seats_avl = [s.text for s in avl_seats]

        # Trim lists to the minimum length to avoid mismatch
        min_len = min(len(companies), len(dept_time), len(arr_time), len(fare), len(seats_avl))
        df = pd.DataFrame({
            "Bus Name": companies[:min_len],
            "Departure Time": dept_time[:min_len],
            "Arrival Time": arr_time[:min_len],
            "Fare": fare[:min_len],
            "Seat Availability": seats_avl[:min_len],
            "Booking Link": [updated_url] * min_len
        })

        print("✅ Scraping completed successfully.")
        return df

    finally:
        driver.quit()