#!/bin/bash

# Quick deployment script for mask detection web interface

echo "🚀 Mask Detection Web Interface Deployment"
echo "=========================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Mask detection web interface"
fi

echo ""
echo "🌐 Choose deployment platform:"
echo "1) GitHub Pages (Free)"
echo "2) Netlify (Recommended)"
echo "3) Vercel (Fast)"
echo "4) Firebase (Google)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "📖 GitHub Pages selected"
        echo "1. Create repository on GitHub"
        echo "2. Push your code:"
        echo "   git remote add origin https://github.com/yourusername/mask-detection-web.git"
        echo "   git branch -M main"
        echo "   git push -u origin main"
        echo "3. Enable Pages in repository settings"
        echo "4. Your site will be at: https://yourusername.github.io/mask-detection-web/"
        ;;
    2)
        echo "🎯 Netlify selected"
        echo "1. Go to https://netlify.com"
        echo "2. Drag & drop your mask_detection_web.html file"
        echo "3. Or connect your GitHub repository"
        echo "4. Get instant URL with custom domain support"
        ;;
    3)
        echo "⚡ Vercel selected"
        echo "Installing Vercel CLI..."
        if command -v npm &> /dev/null; then
            npm i -g vercel
            echo "Run: vercel"
            echo "Follow the prompts to deploy"
        else
            echo "Please install Node.js first, then run: npm i -g vercel"
        fi
        ;;
    4)
        echo "🔥 Firebase selected"
        echo "Installing Firebase CLI..."
        if command -v npm &> /dev/null; then
            npm i -g firebase-tools
            echo "Run: firebase login"
            echo "Then: firebase init hosting"
            echo "Finally: firebase deploy"
        else
            echo "Please install Node.js first, then run: npm i -g firebase-tools"
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        ;;
esac

echo ""
echo "✅ Check the deployment guides for detailed instructions:"
echo "   - deploy-github-pages.md"
echo "   - deploy-netlify.md" 
echo "   - deploy-vercel.md"
echo "   - deploy-firebase.md"
