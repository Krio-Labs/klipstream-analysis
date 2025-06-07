"""
Subprocess Wrapper for FastAPI Environment

This module provides a wrapper for subprocess execution that ensures proper
environment variable inheritance and working directory context when running
in FastAPI thread pool executors.
"""

import os
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform

logger = logging.getLogger(__name__)


class FastAPISubprocessWrapper:
    """
    Wrapper for subprocess execution in FastAPI environment
    
    This class ensures that subprocess execution works correctly when called
    from FastAPI background tasks and thread pool executors.
    """
    
    def __init__(self):
        self.base_env = self._prepare_base_environment()
        self.working_directory = self._get_working_directory()
        
    def _prepare_base_environment(self) -> Dict[str, str]:
        """Prepare base environment variables for subprocess execution"""
        env = os.environ.copy()
        
        # Ensure .NET bundle extraction directory is set
        if platform.system() == "Darwin":  # macOS
            dotnet_dir = Path.home() / '.dotnet' / 'bundle_extract'
        else:  # Linux (Cloud Run)
            dotnet_dir = Path('/tmp/.dotnet/bundle_extract')
        
        # Create directory if it doesn't exist
        dotnet_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable
        env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(dotnet_dir)
        
        # Ensure PATH includes binary directories
        binary_dir = Path(__file__).parent.parent.parent / 'raw_pipeline' / 'bin'
        if binary_dir.exists():
            current_path = env.get('PATH', '')
            if str(binary_dir) not in current_path:
                env['PATH'] = f"{binary_dir}:{current_path}"
        
        return env
    
    def _get_working_directory(self) -> Path:
        """Get the correct working directory for subprocess execution"""
        # Always use the project root directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        return project_root
    
    async def run_subprocess_async(
        self,
        command: List[str],
        timeout: Optional[int] = None,
        capture_output: bool = True,
        check: bool = True,
        additional_env: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        """
        Run subprocess asynchronously with proper environment setup
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit
            additional_env: Additional environment variables
            
        Returns:
            CompletedProcess result
        """
        # Prepare environment
        env = self.base_env.copy()
        if additional_env:
            env.update(additional_env)
        
        # Command execution (logging disabled for cleaner output)
        
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        def run_subprocess():
            return subprocess.run(
                command,
                cwd=self.working_directory,
                env=env,
                timeout=timeout,
                capture_output=capture_output,
                check=check,
                text=True
            )
        
        try:
            result = await loop.run_in_executor(None, run_subprocess)
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Unexpected error running command: {str(e)}")
            raise
    
    def run_subprocess_sync(
        self,
        command: List[str],
        timeout: Optional[int] = None,
        capture_output: bool = True,
        check: bool = True,
        additional_env: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        """
        Run subprocess synchronously with proper environment setup
        
        This is for use in synchronous contexts or when called from
        thread pool executors.
        """
        # Prepare environment
        env = self.base_env.copy()
        if additional_env:
            env.update(additional_env)
        
        # Command execution (logging disabled for cleaner output)
        
        try:
            result = subprocess.run(
                command,
                cwd=self.working_directory,
                env=env,
                timeout=timeout,
                capture_output=capture_output,
                check=check,
                text=True
            )
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Unexpected error running command: {str(e)}")
            raise
    
    def test_twitch_cli(self) -> bool:
        """
        Test if TwitchDownloaderCLI is working properly
        
        Returns:
            True if CLI is working, False otherwise
        """
        try:
            from utils.config import BINARY_PATHS
            
            cli_path = BINARY_PATHS["twitch_downloader"]
            # Test with --help command
            result = self.run_subprocess_sync(
                [cli_path, "--help"],
                timeout=10,
                capture_output=True,
                check=False  # Don't raise exception, just check return code
            )

            # TwitchDownloaderCLI returns 1 for --help, which is normal
            return result.returncode in [0, 1]
                
        except Exception:
            return False


# Global instance
subprocess_wrapper = FastAPISubprocessWrapper()
