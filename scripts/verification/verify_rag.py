#!/usr/bin/env python3
"""
RAG System Verification Script
Checks all dependencies and FAISS index for the Insurance Agent Analytics Platform
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all RAG dependencies are installed"""
    print("🔍 Checking RAG Dependencies...\n")
    
    dependencies = {
        'langchain-community': 'from langchain_community.vectorstores import FAISS',
        'langchain embeddings': 'from langchain_community.embeddings import HuggingFaceEmbeddings',
        'faiss-cpu': 'import faiss',
        'sentence-transformers': 'from sentence_transformers import SentenceTransformer',
        'pandas': 'import pandas as pd',
        'streamlit': 'import streamlit as st'
    }
    
    missing = []
    installed = []
    
    for name, import_stmt in dependencies.items():
        try:
            exec(import_stmt)
            print(f"✅ {name}")
            installed.append(name)
        except ImportError as e:
            print(f"❌ {name} - {str(e)}")
            missing.append(name)
    
    print(f"\n📊 Summary: {len(installed)}/{len(dependencies)} dependencies installed")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\n🔧 To install missing dependencies:")
        print("   python3 -m pip install langchain-community langchain-huggingface faiss-cpu sentence-transformers")
        return False
    else:
        print("\n🎉 All dependencies installed!")
        return True

def check_faiss_index():
    """Check if FAISS index exists and is loadable"""
    print("\n\n🔍 Checking FAISS Index...\n")
    
    # Check for index files
    possible_paths = [
        Path('enhanced_faiss_index'),
        Path('Automobile/enhanced_faiss_index'),
        Path('../enhanced_faiss_index')
    ]
    
    index_path = None
    for path in possible_paths:
        if path.exists():
            index_path = path
            print(f"✅ Found index at: {path.absolute()}")
            break
    
    if not index_path:
        print("❌ FAISS index directory not found")
        print("\n📍 Searched in:")
        for path in possible_paths:
            print(f"   - {path.absolute()}")
        print("\n🔧 Make sure enhanced_faiss_index/ exists in the Automobile folder")
        return False
    
    # Check index files
    index_faiss = index_path / 'index.faiss'
    index_pkl = index_path / 'index.pkl'
    
    if not index_faiss.exists():
        print(f"❌ Missing: {index_faiss}")
        return False
    else:
        size_mb = index_faiss.stat().st_size / (1024*1024)
        print(f"✅ index.faiss ({size_mb:.1f} MB)")
    
    if not index_pkl.exists():
        print(f"❌ Missing: {index_pkl}")
        return False
    else:
        size_mb = index_pkl.stat().st_size / (1024*1024)
        print(f"✅ index.pkl ({size_mb:.1f} MB)")
    
    # Try to load index
    try:
        print("\n📦 Loading embeddings model (may take 30 seconds)...")
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("📊 Loading FAISS index...")
        faiss_db = FAISS.load_local(
            str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ FAISS index loaded successfully!")
        print(f"📈 Total documents indexed: {faiss_db.index.ntotal:,}")
        
        # Test search
        print("\n🔍 Testing search functionality...")
        results = faiss_db.similarity_search("high churn risk customer", k=3)
        print(f"✅ Search successful - returned {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {str(e)}")
        return False

def check_data_file():
    """Check if rag_model_predictions.csv exists"""
    print("\n\n🔍 Checking Data File...\n")
    
    data_file = Path('rag_model_predictions.csv')
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file.absolute()}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(data_file)
        size_mb = data_file.stat().st_size / (1024*1024)
        print(f"✅ rag_model_predictions.csv ({size_mb:.1f} MB)")
        print(f"📊 {len(df):,} customer records loaded")
        print(f"📋 {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"❌ Failed to load data file: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("🤖 RAG System Verification for Insurance Agent Analytics")
    print("="*60)
    
    deps_ok = check_dependencies()
    index_ok = check_faiss_index() if deps_ok else False
    data_ok = check_data_file()
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    status = []
    status.append(("Dependencies", deps_ok))
    status.append(("FAISS Index", index_ok))
    status.append(("Data File", data_ok))
    
    for item, ok in status:
        symbol = "✅" if ok else "❌"
        print(f"{symbol} {item}")
    
    if all(ok for _, ok in status):
        print("\n🎉 SUCCESS! RAG system is ready to use!")
        print("\n▶️  Next step:")
        print("   streamlit run app.py")
        print("   Then navigate to: 🤖 AI Customer Assistant")
        return 0
    else:
        print("\n⚠️  Some components need attention - see details above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
