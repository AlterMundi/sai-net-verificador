#!/bin/bash
# Manual FIgLib extraction script
# Run this after downloading the dataset manually

echo "🔍 Looking for downloaded FIgLib archives..."

# Check for zip file
if [ -f "data/real_figlib/HPWREN-FIgLib.zip" ]; then
    echo "📦 Extracting ZIP archive..."
    cd data/real_figlib
    unzip -q HPWREN-FIgLib.zip
    echo "✅ Extraction complete!"
    
# Check for tar.gz file
elif [ -f "data/real_figlib/HPWREN-FIgLib.tar.gz" ]; then
    echo "📦 Extracting TAR.GZ archive..."
    cd data/real_figlib
    tar -xzf HPWREN-FIgLib.tar.gz
    echo "✅ Extraction complete!"
    
else
    echo "❌ No FIgLib archive found in data/real_figlib/"
    echo "Please download manually first!"
fi

# Count images
echo ""
echo "📊 Dataset Statistics:"
find data/real_figlib -name "*.jpg" -o -name "*.jpeg" | wc -l | xargs echo "Total images:"
