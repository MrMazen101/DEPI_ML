import sys
import webbrowser
import time
import pyautogui

# هنا بنقول للكود: لو في لينك جالك من الـ n8n، خده استخدمه
if len(sys.argv) > 1:
    teams_url = sys.argv[1]
else:
    print("Error: No link provided!")
    sys.exit() # لو مفيش لينك، اقفل السكريبت

print(f"Opening Teams with link: {teams_url}")
webbrowser.open(teams_url)

# هيستنى 15 ثانية لحد ما تيمز يفتح
time.sleep(15)

# هيدوس Tab عشان يروح لزرار Join وبعدين يدوس Enter
pyautogui.press('tab')
time.sleep(1)
pyautogui.press('enter')