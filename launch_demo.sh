#!/bin/bash

# LUMIMAP iPad Interactive Demo - Setup & Launch
# =============================================
# This script installs dependencies and launches the interactive demo

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     🔬 LUMIMAP Interactive Demo - iPad Version           ║"
echo "║     Setup & Launch                                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Install Streamlit
echo "📦 Installing Streamlit (web framework)..."
pip install streamlit --break-system-packages 2>/dev/null || pip install streamlit
echo "✓ Streamlit installed"
echo ""

# Step 2: Check other dependencies
echo "📦 Checking other dependencies..."
pip install torch torchvision pandas numpy matplotlib pillow --break-system-packages 2>/dev/null || \
pip install torch torchvision pandas numpy matplotlib pillow
echo "✓ Dependencies ready"
echo ""

# Step 3: Show startup instructions
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           📱 IPAD CONNECTION INSTRUCTIONS                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Your MacBook IP Address:"
echo ""
ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print "  👉 http://"$2":8501"}'
echo ""
echo "To connect from iPad:"
echo "  1. Go to Safari on iPad"
echo "  2. Type the address above (replace IP with your MacBook IP)"
echo "  3. Hit Enter"
echo "  4. You'll see the interactive demo!"
echo ""

# Step 4: Launch the app
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           🚀 LAUNCHING LUMIMAP INTERACTIVE DEMO           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "The app will start in a few seconds..."
echo "Press Ctrl+C to stop the app when done."
echo ""
sleep 2

# Launch
cd ~/harshu-repo/sfproject
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --logger.level=error
