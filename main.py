#!/usr/bin/env python3
"""
Grok Model Evaluation Suite
Evaluate and compare different Grok models using Grok-4 as a judge
"""

import os
import sys
from dotenv import load_dotenv
from colorama import init, Fore, Style
from dashboard_app import EnhancedEvaluationDashboardEvaluationDashboard
from config import Config

# Initialize colorama for colored output
init(autoreset=True)

def print_banner():
    """Print a nice banner"""
    banner = f"""
{Fore.CYAN}{'='*60}
{Fore.GREEN}🚀 GROK MODEL EVALUATION SUITE {Fore.CYAN}v1.0
{Fore.YELLOW}Evaluate and compare Grok models with interactive dashboard
{Fore.CYAN}{'='*60}{Style.RESET_ALL}
    """
    print(banner)

def check_api_key():
    """Check if API key is configured"""
    if not Config.XAI_API_KEY:
        print(f"{Fore.RED}❌ Error: XAI_API_KEY not found!{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Please create a .env file with:{Style.RESET_ALL}")
        print("XAI_API_KEY=your_api_key_here")
        print("\nOr set it as an environment variable:")
        print("export XAI_API_KEY=your_api_key_here")
        return False
    return True

def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Print banner
    print_banner()
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    print(f"{Fore.GREEN}✓ API Key configured{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Starting dashboard on port {Config.DASHBOARD_PORT}...{Style.RESET_ALL}\n")
    
    try:
        # Create and run dashboard
        dashboard = EvaluationDashboard(Config.XAI_API_KEY)
        
        print(f"{Fore.GREEN}✓ Dashboard initialized successfully{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}➜ Open your browser at: {Fore.CYAN}http://localhost:{Config.DASHBOARD_PORT}{Style.RESET_ALL}")
        print(f"\n{Fore.MAGENTA}Press Ctrl+C to stop the server{Style.RESET_ALL}\n")
        
        # Run the dashboard
        dashboard.run(debug=Config.DEBUG_MODE, port=Config.DASHBOARD_PORT)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutting down gracefully...{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()