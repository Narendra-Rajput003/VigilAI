"""
Comprehensive Test Runner for VigilAI
Runs all test suites and generates comprehensive reports
"""

import pytest
import asyncio
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import subprocess
import coverage
import xmlrunner
from pathlib import Path

class VigilAITestRunner:
    """Comprehensive test runner for VigilAI system"""
    
    def __init__(self):
        self.test_results = {}
        self.coverage_data = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report"""
        print("🚀 Starting VigilAI Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = datetime.now()
        
        # Test suites to run
        test_suites = [
            {
                "name": "Phase 1 - Prototype Tests",
                "file": "test_phase1.py",
                "description": "MVP prototype functionality"
            },
            {
                "name": "Phase 2 - Core Development Tests", 
                "file": "test_phase2.py",
                "description": "AI/ML models and inference"
            },
            {
                "name": "Phase 3 - Cloud Backend Tests",
                "file": "test_phase3_cloud_backend.py", 
                "description": "Cloud infrastructure and streaming"
            },
            {
                "name": "Phase 4 - Deployment Tests",
                "file": "test_phase4_deployment.py",
                "description": "Production deployment and monitoring"
            },
            {
                "name": "Basic System Tests",
                "file": "test_basic.py",
                "description": "Basic system functionality"
            }
        ]
        
        # Run each test suite
        for suite in test_suites:
            print(f"\n📋 Running {suite['name']}...")
            print(f"   Description: {suite['description']}")
            
            result = self._run_test_suite(suite)
            self.test_results[suite['name']] = result
            
            # Print immediate results
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"   Result: {status} ({result['passed_count']}/{result['total_count']} tests)")
            
            if result['failed_tests']:
                print(f"   Failed tests: {', '.join(result['failed_tests'])}")
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Save report to file
        self._save_report(report)
        
        return report
    
    def _run_test_suite(self, suite: Dict[str, str]) -> Dict[str, Any]:
        """Run individual test suite"""
        test_file = Path(__file__).parent / suite['file']
        
        if not test_file.exists():
            return {
                'passed': False,
                'total_count': 0,
                'passed_count': 0,
                'failed_count': 0,
                'failed_tests': [],
                'error': f"Test file not found: {suite['file']}"
            }
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                str(test_file),
                '-v', '--tb=short', '--junitxml=f"test_results_{suite["name"].lower().replace(" ", "_")}.xml"'
            ], capture_output=True, text=True, timeout=300)
            
            # Parse results
            passed_count = result.stdout.count('PASSED')
            failed_count = result.stdout.count('FAILED')
            total_count = passed_count + failed_count
            
            # Extract failed test names
            failed_tests = []
            if failed_count > 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'FAILED' in line:
                        test_name = line.split('::')[-1].split(' ')[0]
                        failed_tests.append(test_name)
            
            return {
                'passed': failed_count == 0,
                'total_count': total_count,
                'passed_count': passed_count,
                'failed_count': failed_count,
                'failed_tests': failed_tests,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'total_count': 0,
                'passed_count': 0,
                'failed_count': 0,
                'failed_tests': [],
                'error': 'Test suite timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'passed': False,
                'total_count': 0,
                'passed_count': 0,
                'failed_count': 0,
                'failed_tests': [],
                'error': str(e)
            }
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = sum(result['total_count'] for result in self.test_results.values())
        total_passed = sum(result['passed_count'] for result in self.test_results.values())
        total_failed = sum(result['failed_count'] for result in self.test_results.values())
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        overall_status = "✅ ALL TESTS PASSED" if total_failed == 0 else "❌ SOME TESTS FAILED"
        
        # Generate report
        report = {
            'test_run_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': duration,
                'overall_status': overall_status
            },
            'summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'success_rate_percent': round(success_rate, 2)
            },
            'test_suites': self.test_results,
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for suite_name, result in self.test_results.items():
            if not result['passed']:
                if 'Phase 1' in suite_name:
                    recommendations.append("🔧 Fix Phase 1 prototype issues before proceeding")
                elif 'Phase 2' in suite_name:
                    recommendations.append("🤖 Review AI/ML model implementations")
                elif 'Phase 3' in suite_name:
                    recommendations.append("☁️ Check cloud infrastructure and streaming setup")
                elif 'Phase 4' in suite_name:
                    recommendations.append("📊 Verify monitoring and analytics configuration")
        
        if not recommendations:
            recommendations.append("🎉 All tests passed! System is ready for production deployment.")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on test results"""
        next_steps = []
        
        # Check if all phases passed
        all_phases_passed = all(
            result['passed'] for name, result in self.test_results.items() 
            if 'Phase' in name
        )
        
        if all_phases_passed:
            next_steps.extend([
                "🚀 Deploy to production environment",
                "📈 Set up monitoring and alerting",
                "👥 Onboard beta users",
                "📊 Monitor system performance",
                "🔄 Implement continuous integration"
            ])
        else:
            next_steps.extend([
                "🔍 Review failed test cases",
                "🛠️ Fix identified issues",
                "🧪 Re-run test suite",
                "📝 Update documentation",
                "🔄 Iterate and improve"
            ])
        
        return next_steps
    
    def _save_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 VIGILAI TEST SUMMARY")
        print("=" * 60)
        
        print(f"⏱️  Duration: {report['test_run_info']['duration_seconds']:.2f} seconds")
        print(f"📈 Status: {report['test_run_info']['overall_status']}")
        print(f"🧪 Total Tests: {report['summary']['total_tests']}")
        print(f"✅ Passed: {report['summary']['passed_tests']}")
        print(f"❌ Failed: {report['summary']['failed_tests']}")
        print(f"📊 Success Rate: {report['summary']['success_rate_percent']}%")
        
        print("\n📋 TEST SUITE RESULTS:")
        for suite_name, result in report['test_suites'].items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"   {suite_name}: {status} ({result['passed_count']}/{result['total_count']})")
        
        print("\n💡 RECOMMENDATIONS:")
        for recommendation in report['recommendations']:
            print(f"   {recommendation}")
        
        print("\n🎯 NEXT STEPS:")
        for step in report['next_steps']:
            print(f"   {step}")
        
        print("\n" + "=" * 60)

def main():
    """Main entry point for test runner"""
    runner = VigilAITestRunner()
    
    try:
        report = runner.run_all_tests()
        runner.print_summary(report)
        
        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
