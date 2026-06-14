# 📱 LUMIMAP Interactive Demo for iPad

## What You Have

**`streamlit_app.py`** — The complete interactive app that judges use on iPad!

This is the EXACT same analysis and output as `demo_with_gradcam.py` but:
- ✅ Judges pick the sample (3 options)
- ✅ Judges pick the drug being tested
- ✅ Judges click "Analyze"
- ✅ See exact same beautiful output with all visualizations

---

## 🚀 How to Launch (5 minutes)

### Step 1: Copy files to your project
```bash
cp streamlit_app.py ~/harshu-repo/sfproject/
cp launch_demo.sh ~/harshu-repo/sfproject/
chmod +x ~/harshu-repo/sfproject/launch_demo.sh
```

### Step 2: Install Streamlit (one time)
```bash
pip install streamlit --break-system-packages
```

### Step 3: Launch the app
```bash
cd ~/harshu-repo/sfproject
./launch_demo.sh
```

**That's it!** The app starts running.

### Step 4: Connect iPad

When the script runs, it shows:
```
👉 http://192.168.1.45:8501   (your actual IP will be different)
```

**On your iPad:**
1. Open Safari
2. Type that address (use your actual IP!)
3. Hit Enter
4. You see LUMIMAP on iPad! 🎉

---

## 📱 What Judges See

### Header
- LUMIMAP branding with "🏆 ILC Finalist"
- Professional, beautiful design

### Step 1: Choose Patient Sample
**3 large buttons to tap:**
```
🔴 Sample 36          🟠 Sample 41          🟢 Sample 100
Latrunculin B         Latrunculin B         5-Fluorouracil
(Actin Disruptor)     (Actin Disruptor)     (DNA Replication)
```
Judges tap one → it gets highlighted ✓

### Step 2: Select Drug Being Tested
- Dropdown to choose drug (5-Fluorouracil, Paclitaxel, etc.)
- MOA auto-fills based on drug selection
- Concentration slider

### Step 3: Run Analysis
**Big green button:**
```
🔬 ANALYZE CELL IMAGES
```

### Results (appear instantly)
1. **Classification Banner** — Giant colored box showing:
   - ✅ SENSITIVE (green)
   - ⚠️ PARTIAL_RESISTANCE (orange)
   - 🔄 CROSS_RESISTANCE (red)
   - ⛔ PRIMARY_RESISTANCE (purple)

2. **AI Explanation** — Plain English explaining what AI found

3. **4 Microscopy Images**
   - DAPI (blue) — Nucleus
   - Tubulin (green) — Microtubules
   - Actin (red) — Cytoskeleton
   - GradCAM (orange heatmap) — Where AI looked

4. **Similarity Score Bar Chart**
   - How closely cells match different patterns
   - Green line = threshold for "working"
   - Orange line = partial resistance threshold

5. **Channel Attention Pie Chart**
   - Which cellular structures AI focused on
   - Percentages for each channel

6. **Treatment Recommendation Box**
   - Action to take (continue, adjust, switch, etc.)
   - Specific alternative drugs if needed

7. **Complete Score Table**
   - All numbers with explanations

---

## 💡 What Makes This Perfect for ILC

### Judges Get Full Experience
- "Let me try with this sample and drug..."
- Click "Analyze"
- See beautiful results
- Understand exactly what AI did

### Explains Everything
- Why it made that decision
- Which parts of the cell it looked at
- What to do clinically
- Alternative options

### Real Data
- Uses actual BBBC021 dataset
- Same 3 samples from your science fair demo
- Same AI model they trained

### Professional Presentation
- Polished iPad interface
- Clear sections
- Beautiful visualizations
- Medical design aesthetic

---

## 📋 What's Displayed

### Always Shows:
✅ Classification (SENSITIVE, PARTIAL, CROSS, PRIMARY)
✅ 4 microscopy images with GradCAM
✅ Similarity scores (bar chart)
✅ Attention weights (pie chart)
✅ Treatment recommendation
✅ Clinical interpretation

### Customizable By Judge:
- Choose patient sample (3 options)
- Choose drug being tested
- Set concentration

### AI Provides:
- Resistance classification
- Predicted alternative drugs
- Confidence scores
- Explanation of reasoning

---

## 🎤 Demo Patter for Judges

> **"Welcome! You're going to run the LUMIMAP AI yourself.**
>
> **Please select a patient sample — we have three examples from real breast cancer cells. Then choose what drug we're testing. Finally, hit 'Analyze' and the AI will tell you if it will work and what to recommend instead if it doesn't."**

Judge taps sample → taps drug → hits Analyze → **Results appear in 5 seconds!**

> **"See this? The AI found the cells are RESISTANT to this drug. But look at the attention chart — it focused on the actin cytoskeleton. And the similarity scores show these cells actually respond like they would to a different mechanism — taxanes instead of this actin disruptor."**

> **"That's cross-resistance, and the AI recommended we switch drugs. The entire analysis with all these visualizations happens in 20 minutes. Traditional methods take 2-3 weeks."**

---

## 🔧 Troubleshooting

### "Can't connect from iPad"
- Check iPad and Mac are on same WiFi
- Try typing the IP address slowly
- Make sure you copied the FULL address (http://...)
- Check firewall isn't blocking port 8501

### "Button doesn't work"
- Refresh Safari
- Try tapping again (slight delay is normal)
- Check MacBook terminal — see any error messages?

### "Images don't load"
- Check CSV_PATH and DATA_DIR point to correct locations
- Make sure BBBC021_v1_image.csv exists in project
- Make sure ./data/ folder has images

### "Model won't load"
- Check MODEL_PATH is correct
- Make sure trained model file exists
- Try: `ls ./output/phase1_strategic/`

---

## 📱 Pro Tips for iPad Demo

### Before Demo Starts:
1. Launch app first (5 min before)
2. Test on your MacBook first in browser
3. Have iPad ready with Safari open
4. Know your IP address!

### During Demo:
1. Let judges tap the buttons themselves
2. Let them choose drug (don't pick for them)
3. Let them hit "Analyze" (suspenseful!)
4. Scroll down together to see all results
5. Explain each visualization as you scroll

### If Something Goes Wrong:
1. Have your laptop showing demo in browser as backup
2. Can still talk through what results mean
3. iPad connection usually works — refresh if needed

---

## 🎯 What You're Showing Judges

**Three things:**
1. ✅ **Real System** — Works with actual cancer cell images
2. ✅ **Fast Analysis** — Results in 20 minutes vs 2-3 weeks
3. ✅ **Clinical Value** — Gives specific drug recommendations

**They control it** — makes it impressive that it works so fast!

---

## 📊 Sample Outputs Expected

### Sample 36 (Actin Disruptor)
- Classification: Usually **CROSS_RESISTANCE**
- AI says: "This drug isn't working. But these cells respond to a different mechanism"
- Recommendation: "Switch to Taxanes"

### Sample 41 (Actin Disruptor)
- Classification: Usually **PARTIAL_RESISTANCE**
- AI says: "Some resistance developing"
- Recommendation: "Increase dose or add combination therapy"

### Sample 100 (DNA Replication)
- Classification: Usually **SENSITIVE**
- AI says: "Drug is working normally"
- Recommendation: "Continue current treatment"

---

## 🏆 This is Your ILC Winning Move

Judges get to:
- ✅ Pick what they want to see
- ✅ See it analyze in real-time
- ✅ Understand the AI reasoning
- ✅ Realize clinical impact
- ✅ Experience the speed advantage

**Interactive demos beat static posters!**

---

## ✨ Next Steps

1. Copy files to project
2. Run `./launch_demo.sh`
3. Test on iPad from your own Mac first
4. Practice demo once
5. **You're ready for ILC!** 🏆

---

<p align="center">
  <strong>The judges are going to be IMPRESSED! 🚀</strong>
</p>
