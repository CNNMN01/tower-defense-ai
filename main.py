#!/usr/bin/env python3
"""
Tower Defense AI - Main Entry Point
Choose between lightweight or deep AI versions.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def show_menu():
    """Display the main menu options"""
    print("TOWER DEFENSE AI")
    print("=" * 25)
    print("1. Lightweight AI (Mobile)")
    print("2. Deep AI (Desktop)")
    print("3. Exit")
    print("=" * 25)

def main():
    """Main application entry point"""
    
    while True:
        show_menu()
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == "1":
            print("\nLoading Lightweight AI...")
            try:
                from lightweight_tower_ai import run_lightweight_tower_defense
                run_lightweight_tower_defense()
            except ImportError:
                print("Error: lightweight_tower_ai.py not found in src/")
            except Exception as e:
                print(f"Error running lightweight AI: {e}")
                
        elif choice == "2":
            print("\nLoading Deep AI...")
            try:
                from deep_tower_ai import run_deep_tower_defense_lab
                run_deep_tower_defense_lab()
            except ImportError:
                print("Error: deep_tower_ai.py not found in src/")
            except MemoryError:
                print("Not enough memory for deep AI. Try lightweight version.")
            except Exception as e:
                print(f"Error running deep AI: {e}")
                
        elif choice == "3":
            print("Thanks for playing!")
            break
            
        else:
            print("Invalid choice. Please try again.")
        
        if choice in ["1", "2"]:
            input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()