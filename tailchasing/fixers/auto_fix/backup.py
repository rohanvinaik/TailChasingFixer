"""
Backup management for the auto-fix engine.

Provides safe file backup and restoration capabilities with automatic cleanup.
"""

import hashlib
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages file backups and rollback operations."""
    
    def __init__(self, backup_dir: Optional[str] = None):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Custom backup directory. If None, uses temp directory.
        """
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = Path(tempfile.gettempdir()) / "tailchasing_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_registry: Dict[str, str] = {}
        
    def create_backup(self, file_path: str) -> str:
        """
        Create a backup of the specified file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file
            
        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique backup name
        timestamp = int(time.time())
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        backup_name = f"{file_path.name}_{timestamp}_{file_hash}.backup"
        backup_path = self.backup_dir / backup_name
        
        # Copy file to backup location
        shutil.copy2(file_path, backup_path)
        self.backup_registry[str(file_path)] = str(backup_path)
        
        logger.info(f"Created backup: {file_path} -> {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, file_path: str) -> bool:
        """
        Restore a file from its backup.
        
        Args:
            file_path: Path to file to restore
            
        Returns:
            True if restoration successful, False otherwise
        """
        backup_path = self.backup_registry.get(str(file_path))
        if not backup_path or not Path(backup_path).exists():
            logger.error(f"No backup found for {file_path}")
            return False
        
        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"Restored backup: {backup_path} -> {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup for {file_path}: {e}")
            return False
    
    def has_backup(self, file_path: str) -> bool:
        """
        Check if a backup exists for the specified file.
        
        Args:
            file_path: Path to check for backup
            
        Returns:
            True if backup exists, False otherwise
        """
        backup_path = self.backup_registry.get(str(file_path))
        return backup_path is not None and Path(backup_path).exists()
    
    def list_backups(self) -> Dict[str, str]:
        """
        Get list of all current backups.
        
        Returns:
            Dictionary mapping original file paths to backup paths
        """
        return self.backup_registry.copy()
    
    def cleanup_backups(self, max_age_hours: int = 24) -> int:
        """
        Clean up old backup files.
        
        Args:
            max_age_hours: Maximum age in hours before backup is deleted
            
        Returns:
            Number of backups cleaned up
        """
        current_time = time.time()
        cleaned_count = 0
        
        for backup_file in self.backup_dir.glob("*.backup"):
            if current_time - backup_file.stat().st_mtime > max_age_hours * 3600:
                try:
                    backup_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old backup: {backup_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up backup {backup_file}: {e}")
        
        # Clean up registry entries for non-existent backups
        to_remove = []
        for original, backup_path in self.backup_registry.items():
            if not Path(backup_path).exists():
                to_remove.append(original)
        
        for original in to_remove:
            del self.backup_registry[original]
        
        return cleaned_count
    
    def get_backup_size(self) -> int:
        """
        Get total size of all backups in bytes.
        
        Returns:
            Total size of backup directory in bytes
        """
        total_size = 0
        for backup_file in self.backup_dir.glob("*.backup"):
            total_size += backup_file.stat().st_size
        return total_size


__all__ = ['BackupManager']