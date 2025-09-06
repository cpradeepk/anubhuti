#!/bin/bash

# YAMNet Speech Classification Pipeline - Update Script for Raspberry Pi
# This script handles updates to the deployed system via git pull

set -e  # Exit on any error

# Configuration
INSTALL_DIR="$HOME/anubhuti"
VENV_DIR="$INSTALL_DIR/yamnet_implementation/yamnet_env"
LOG_FILE="$HOME/yamnet_update.log"
BACKUP_DIR="$HOME/anubhuti_backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if deployment exists
check_deployment() {
    log "🔍 Checking existing deployment..."
    
    if [[ ! -d "$INSTALL_DIR" ]]; then
        error "❌ YAMNet deployment not found at $INSTALL_DIR. Run deploy.sh first."
    fi
    
    if [[ ! -d "$INSTALL_DIR/.git" ]]; then
        error "❌ Not a git repository. Cannot update."
    fi
    
    if [[ ! -d "$VENV_DIR" ]]; then
        error "❌ Virtual environment not found. Run deploy.sh first."
    fi
    
    log "✅ Existing deployment found"
}

# Create backup before update
create_backup() {
    log "💾 Creating backup before update..."
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Create timestamped backup
    local backup_name="anubhuti_backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    # Copy current installation
    cp -r "$INSTALL_DIR" "$backup_path" || error "Failed to create backup"
    
    # Remove git directory from backup to save space
    rm -rf "$backup_path/.git" 2>/dev/null || true
    
    # Create backup info file
    cat > "$backup_path/backup_info.txt" << EOF
YAMNet Backup Information
========================

Backup Date: $(date)
Original Path: $INSTALL_DIR
Git Commit: $(cd "$INSTALL_DIR" && git rev-parse HEAD)
Git Branch: $(cd "$INSTALL_DIR" && git branch --show-current)

To restore this backup:
1. Stop any running YAMNet processes
2. Remove current installation: rm -rf $INSTALL_DIR
3. Restore backup: cp -r $backup_path $INSTALL_DIR
4. Reactivate virtual environment: source $VENV_DIR/bin/activate
EOF
    
    log "✅ Backup created: $backup_path"
    echo "$backup_path" > "$HOME/.yamnet_last_backup"
    
    # Clean up old backups (keep last 5)
    local backup_count=$(ls -1 "$BACKUP_DIR" | grep "anubhuti_backup_" | wc -l)
    if [[ $backup_count -gt 5 ]]; then
        info "🧹 Cleaning up old backups (keeping last 5)..."
        ls -1t "$BACKUP_DIR" | grep "anubhuti_backup_" | tail -n +6 | while read old_backup; do
            rm -rf "$BACKUP_DIR/$old_backup"
            info "Removed old backup: $old_backup"
        done
    fi
}

# Check for updates
check_updates() {
    log "🔍 Checking for updates..."
    
    cd "$INSTALL_DIR"
    
    # Fetch latest changes
    git fetch origin main || error "Failed to fetch updates"
    
    # Check if updates are available
    local local_commit=$(git rev-parse HEAD)
    local remote_commit=$(git rev-parse origin/main)
    
    if [[ "$local_commit" == "$remote_commit" ]]; then
        log "✅ Already up to date (commit: ${local_commit:0:8})"
        return 1  # No updates available
    else
        log "📥 Updates available:"
        log "   Current: ${local_commit:0:8}"
        log "   Latest:  ${remote_commit:0:8}"
        
        # Show what will be updated
        info "Changes to be applied:"
        git log --oneline "$local_commit..$remote_commit" | tee -a "$LOG_FILE"
        
        return 0  # Updates available
    fi
}

# Apply updates
apply_updates() {
    log "📥 Applying updates..."
    
    cd "$INSTALL_DIR"
    
    # Pull latest changes
    git pull origin main || error "Failed to pull updates"
    
    local new_commit=$(git rev-parse HEAD)
    log "✅ Updated to commit: ${new_commit:0:8}"
}

# Update Python dependencies
update_dependencies() {
    log "📚 Checking Python dependencies..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    
    # Check if requirements.txt was updated
    if git diff HEAD~1 HEAD --name-only | grep -q "requirements.txt"; then
        log "📦 requirements.txt was updated. Installing new dependencies..."
        
        source yamnet_env/bin/activate || error "Failed to activate virtual environment"
        
        # Update pip first
        pip install --upgrade pip
        
        # Install/update requirements
        if [[ -f "requirements.txt" ]]; then
            pip install -r requirements.txt --upgrade || warning "Some dependencies failed to update"
        fi
        
        log "✅ Dependencies updated"
    else
        log "✅ No dependency updates needed"
    fi
}

# Test updated deployment
test_update() {
    log "🧪 Testing updated deployment..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Test model loading
    python3 -c "
import tensorflow as tf
print('Testing updated model...')
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print('✅ Model loads successfully after update')
print(f'Model parameters: {model.count_params():,}')
" || error "Model test failed after update"
    
    # Test with sample audio if available
    if [[ -f "../slow/Fhmm_slow.wav" ]]; then
        info "Testing with sample audio..."
        python3 test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet || warning "Sample audio test failed"
    fi
    
    log "✅ Update test passed"
}

# Rollback function
rollback() {
    log "🔄 Rolling back to previous version..."
    
    if [[ ! -f "$HOME/.yamnet_last_backup" ]]; then
        error "❌ No backup found for rollback"
    fi
    
    local backup_path=$(cat "$HOME/.yamnet_last_backup")
    
    if [[ ! -d "$backup_path" ]]; then
        error "❌ Backup directory not found: $backup_path"
    fi
    
    # Stop any running processes
    pkill -f "yamnet" 2>/dev/null || true
    
    # Remove current installation
    rm -rf "$INSTALL_DIR"
    
    # Restore backup
    cp -r "$backup_path" "$INSTALL_DIR" || error "Failed to restore backup"
    
    # Reinitialize git repository
    cd "$INSTALL_DIR"
    git init
    git remote add origin https://github.com/cpradeepk/anubhuti.git
    
    log "✅ Rollback completed successfully"
    log "🔄 Restored from backup: $backup_path"
}

# Create update summary
create_update_summary() {
    log "📋 Creating update summary..."
    
    local summary_file="$HOME/yamnet_update_summary.txt"
    local current_commit=$(cd "$INSTALL_DIR" && git rev-parse HEAD)
    
    cat > "$summary_file" << EOF
YAMNet Speech Classification Pipeline - Update Summary
=====================================================

Update Date: $(date)
Installation Directory: $INSTALL_DIR
New Commit: $current_commit
Log File: $LOG_FILE

Recent Changes:
$(cd "$INSTALL_DIR" && git log --oneline -5)

System Status:
- Model File: $(ls -lh $INSTALL_DIR/yamnet_implementation/yamnet_models/yamnet_classifier.h5 | awk '{print $5}')
- Virtual Environment: Active
- Last Backup: $(cat "$HOME/.yamnet_last_backup" 2>/dev/null || echo "None")

Quick Commands:
1. Test deployment: cd $INSTALL_DIR/yamnet_implementation && python3 health_check.py
2. Run classification: python3 test_yamnet_model.py audio_file.wav
3. Real-time mode: python3 realtime_pi_test.py
4. Rollback if needed: $INSTALL_DIR/update.sh --rollback

For issues, check the log file: $LOG_FILE
EOF
    
    log "✅ Update summary created: $summary_file"
}

# Show help
show_help() {
    echo "YAMNet Update Script"
    echo "==================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check-only    Check for updates without applying them"
    echo "  --force         Force update even if no changes detected"
    echo "  --rollback      Rollback to previous backup"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Normal update"
    echo "  $0 --check-only       # Just check for updates"
    echo "  $0 --rollback         # Rollback to previous version"
}

# Main update function
main() {
    local check_only=false
    local force_update=false
    local do_rollback=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                check_only=true
                shift
                ;;
            --force)
                force_update=true
                shift
                ;;
            --rollback)
                do_rollback=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          YAMNet Speech Classification Pipeline               ║"
    echo "║                    Update Script                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Create log file
    touch "$LOG_FILE"
    log "📝 Logging to: $LOG_FILE"
    
    # Handle rollback
    if [[ "$do_rollback" == true ]]; then
        rollback
        exit 0
    fi
    
    # Check existing deployment
    check_deployment
    
    # Check for updates
    if check_updates || [[ "$force_update" == true ]]; then
        if [[ "$check_only" == true ]]; then
            log "✅ Updates are available. Run without --check-only to apply them."
            exit 0
        fi
        
        # Perform update
        create_backup
        apply_updates
        update_dependencies
        test_update
        create_update_summary
        
        echo -e "${GREEN}"
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║                    🎉 UPDATE SUCCESSFUL! 🎉                  ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
        
        log "🎉 YAMNet update completed successfully!"
        log "📋 Check update summary: $HOME/yamnet_update_summary.txt"
        
    else
        if [[ "$check_only" == true ]]; then
            log "✅ No updates available."
        else
            log "✅ Already up to date. No action needed."
        fi
    fi
    
    echo ""
    echo -e "${BLUE}System Status:${NC}"
    echo "- Installation: $INSTALL_DIR"
    echo "- Virtual Environment: $VENV_DIR"
    echo "- Log File: $LOG_FILE"
    echo ""
    echo -e "${GREEN}Your YAMNet system is ready! 🎵🤖✨${NC}"
}

# Handle script interruption
trap 'error "❌ Update interrupted by user"' INT TERM

# Run main function
main "$@"
