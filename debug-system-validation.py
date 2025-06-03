#!/usr/bin/env python3
"""
System validation debug script
"""

import os
import sys
import psutil
from deploy_raptor_production import DeploymentValidator

def debug_system_validation():
    """Debug system validation with detailed output"""
    
    print("🔍 RAPTOR System Validation Debug")
    print("=" * 50)
    
    result = DeploymentValidator.validate_system_requirements()
    
    print(f"Overall Status: {'✅ PASS' if result.success else '❌ FAIL'}")
    print(f"Message: {result.message}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print()
    
    # Detailed breakdown
    checks = result.details.get('checks', {})
    details = result.details.get('details', {})
    
    print("📋 Detailed Check Results:")
    print("-" * 30)
    
    # Memory check
    memory_ok = checks.get('memory', False)
    memory_details = details.get('memory', {})
    print(f"Memory: {'✅ PASS' if memory_ok else '❌ FAIL'}")
    print(f"  Total: {memory_details.get('total_gb', 0):.1f} GB")
    print(f"  Required: {memory_details.get('required_gb', 8)} GB")
    print(f"  Available: {memory_details.get('available_gb', 0):.1f} GB")
    print()
    
    # CPU check
    cpu_ok = checks.get('cpu', False)
    cpu_details = details.get('cpu', {})
    print(f"CPU: {'✅ PASS' if cpu_ok else '❌ FAIL'}")
    print(f"  Cores: {cpu_details.get('cores', 0)}")
    print(f"  Recommended: {cpu_details.get('recommended', 4)}")
    print()
    
    # Disk check
    disk_ok = checks.get('disk', False)
    disk_details = details.get('disk', {})
    print(f"Disk Space: {'✅ PASS' if disk_ok else '❌ FAIL'}")
    print(f"  Free: {disk_details.get('free_gb', 0):.1f} GB")
    print(f"  Required: {disk_details.get('required_gb', 50)} GB")
    print()
    
    # Python check
    python_ok = checks.get('python', False)
    python_details = details.get('python', {})
    print(f"Python Version: {'✅ PASS' if python_ok else '❌ FAIL'}")
    print(f"  Current: {python_details.get('current', 'unknown')}")
    print(f"  Required: {python_details.get('required', '3.8+')}")
    print()
    
    # Package check
    packages_ok = checks.get('packages', False)
    package_details = details.get('packages', {})
    print(f"Required Packages: {'✅ PASS' if packages_ok else '❌ FAIL'}")
    for package, status in package_details.items():
        print(f"  {package}: {'✅' if status else '❌'}")
    print()
    
    # Environment variables check
    env_ok = checks.get('environment', False)
    env_details = details.get('environment', {})
    print(f"Environment Variables: {'✅ PASS' if env_ok else '❌ FAIL'}")
    for var, status in env_details.items():
        print(f"  {var}: {'✅ SET' if status else '❌ MISSING'}")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print("-" * 20)
    
    if not memory_ok:
        print("• Increase system memory to at least 8GB for production")
    
    if not cpu_ok:
        print("• Use a system with at least 4 CPU cores")
    
    if not disk_ok:
        print("• Free up disk space (need at least 50GB)")
    
    if not python_ok:
        print("• Upgrade Python to version 3.8 or newer")
    
    if not packages_ok:
        failed_packages = [pkg for pkg, status in package_details.items() if not status]
        if failed_packages:
            print(f"• Install missing packages: pip install {' '.join(failed_packages)}")
    
    if not env_ok:
        missing_vars = [var for var, status in env_details.items() if not status]
        if missing_vars:
            print(f"• Set missing environment variables: {', '.join(missing_vars)}")
            if 'OPENAI_API_KEY' in missing_vars:
                print("  Create .env file with: OPENAI_API_KEY=your_key_here")
    
    print()
    print("🔧 Quick fixes:")
    print("-" * 15)
    
    # Generate fix commands
    if not packages_ok:
        failed_packages = [pkg for pkg, status in package_details.items() if not status]
        if failed_packages:
            print(f"pip install {' '.join(failed_packages)}")
    
    if not env_ok and 'OPENAI_API_KEY' in [var for var, status in env_details.items() if not status]:
        print("echo 'OPENAI_API_KEY=your_openai_key_here' > .env")
    
    print()
    return result.success

if __name__ == "__main__":
    success = debug_system_validation()
    sys.exit(0 if success else 1)