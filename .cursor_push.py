#!/usr/bin/env python3
"""
Script automatico per push git - bypassa problemi shell Cursor
Esegui: python3 .cursor_push.py
"""
import subprocess
import sys
import os

def run_git_push():
    repo_path = '/Users/metaproos/Desktop/face-site'
    os.chdir(repo_path)
    
    try:
        # Verifica se ci sono modifiche
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            print("ℹ️  Nessuna modifica da committare")
            return
        
        # Add file modificati
        print("📝 Aggiungo modifiche...")
        subprocess.run(['git', 'add', '-A'], check=True)
        
        # Commit
        print("💾 Creo commit...")
        commit_msg = "Fix: Stripe metadata 500 char limit - use token for long photo_ids lists"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        # Push
        print("🚀 Push su GitHub...")
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        # Hash commit
        result = subprocess.run(['git', 'log', '-1', '--format=%H'], 
                              capture_output=True, text=True, check=True)
        commit_hash = result.stdout.strip()
        
        print(f"\n✅ Push completato!")
        print(f"📋 Hash: {commit_hash}")
        print(f"🔗 Verifica su Render con questo hash")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        sys.exit(1)

if __name__ == '__main__':
    run_git_push()
