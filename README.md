# 🏰 Tower Defense AI

An AI that learns to play tower defense and gets better over time. The AI remembers what it learns between sessions.

## 🚀 Quick Start

### Mobile (iOS - Pythonista/Pyto)
```python
exec(open('src/lightweight_tower_ai.py').read())
```

### Desktop/Laptop

```python
exec(open('src/deep_tower_ai.py').read())
```

### Interactive Menu

```python
python main.py
```

## 🧠 How It Works

- AI starts terrible (0% wins)
- Learns through trial and error
- Saves knowledge between sessions
- Gets better each time you run it
- Eventually masters the game (60%+ wins)

## 📊 Two AI Versions

|Version    |Best For|Memory|Learning     |
|-----------|--------|------|-------------|
|Lightweight|Mobile  |50KB  |Fast         |
|Deep       |Desktop |2MB   |Sophisticated|

## 🎮 Game Rules

- Stop enemies from reaching the end
- Build towers to shoot enemies
- Different tower types cost different amounts
- Survive all waves to win

## 📱 Requirements

- Python 3.7+
- NumPy
- Matplotlib (optional)

## 🔧 Installation

```bash
git clone https://github.com/yourusername/tower-defense-ai.git
cd tower-defense-ai
python main.py
```

## 📈 Example Progress

```
Session 1: 0% wins - Random placement
Session 5: 20% wins - Learning basics  
Session 10: 40% wins - Good strategy
Session 20: 60%+ wins - AI mastery
```