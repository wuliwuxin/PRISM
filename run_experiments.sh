#!/bin/bash

################################################################################
# PRISM Optimized Experiment Runner
# Supports multiple prediction modes and flexible configurations
################################################################################

echo "================================================================================"
echo "PRISM: Primitive-based Recurrent Inference for Sequence Modeling"
echo "Optimized Experiment Runner"
echo "================================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
MODE="total"
SEEDS="42 2024 123456 2025 2026"
PRED_LENS="6 12 24 48"
GPUS="0 1"
EPOCHS=100
BATCH_SIZE=128

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --seeds)
            shift
            SEEDS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SEEDS="$SEEDS $1"
                shift
            done
            ;;
        --pred-lens)
            shift
            PRED_LENS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                PRED_LENS="$PRED_LENS $1"
                shift
            done
            ;;
        --gpus)
            shift
            GPUS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                GPUS="$GPUS $1"
                shift
            done
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE          Prediction mode: total/priority/organization (default: total)"
            echo "  --seeds SEED...      Random seeds (default: 42 2024 123456 2025 2026)"
            echo "  --pred-lens LEN...   Prediction lengths (default: 6 12 24 48)"
            echo "  --gpus GPU...        GPU IDs (default: 0 1)"
            echo "  --epochs N           Number of epochs (default: 100)"
            echo "  --batch-size N       Batch size (default: 128)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_experiments.sh --mode total --seeds 42 2024 --pred-lens 24 48"
            echo "  ./run_experiments.sh --mode priority --gpus 0 --epochs 50"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "\n${BLUE}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Seeds: $SEEDS"
echo "  Prediction Lengths: $PRED_LENS"
echo "  GPUs: $GPUS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"
python -c "import torch; import pandas; import numpy; import sklearn; import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Missing required packages${NC}"
    echo "Install: pip install torch pandas numpy scikit-learn tqdm matplotlib seaborn"
    exit 1
fi
echo -e "${GREEN}✓ All dependencies satisfied${NC}"

# Check CUDA
echo -e "\n${YELLOW}Checking CUDA...${NC}"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

# Create directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p checkpoints results visualizations predictions logs data
echo -e "${GREEN}✓ Directories created${NC}"

# Check data files
echo -e "\n${YELLOW}Checking data files...${NC}"
if [ ! -f "data/node_info_df.csv" ] || [ ! -f "data/job_info_df.csv" ]; then
    echo -e "${RED}✗ Data files not found${NC}"
    echo "Required files:"
    echo "  - data/node_info_df.csv"
    echo "  - data/job_info_df.csv"
    exit 1
fi
echo -e "${GREEN}✓ Data files found${NC}"

# Run experiments
echo -e "\n================================================================================"
echo -e "${YELLOW}Starting experiments...${NC}"
echo "This may take several hours"
echo "================================================================================"

START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build command
CMD="python main.py"
CMD="$CMD --mode $MODE"
CMD="$CMD --seeds $SEEDS"
CMD="$CMD --pred_lens $PRED_LENS"
CMD="$CMD --gpus $GPUS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"

echo "Command: $CMD"
echo ""

# Run with logging
$CMD 2>&1 | tee logs/experiment_${MODE}_${TIMESTAMP}.log

EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✓ Experiments completed successfully${NC}"
    echo -e "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"

    # Generate visualizations
    echo -e "\n${YELLOW}Generating visualizations...${NC}"
    python visualize.py --mode $MODE 2>&1 | tee logs/visualization_${MODE}_${TIMESTAMP}.log

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Visualizations generated${NC}"
    else
        echo -e "${YELLOW}⚠ Visualization had warnings${NC}"
    fi

    # Summary
    echo -e "\n================================================================================"
    echo -e "${GREEN}ALL TASKS COMPLETED${NC}"
    echo -e "================================================================================"
    echo "Results:"
    echo "  - results/prism_${MODE}_results.csv"
    echo "  - results/summary_${MODE}.txt"
    echo ""
    echo "Visualizations:"
    echo "  - visualizations/*.png"
    echo ""
    echo "Predictions (.npy files):"
    echo "  - predictions/*_predictions.npy"
    echo "  - predictions/*_targets.npy"
    echo ""
    echo "Models:"
    echo "  - checkpoints/*.pth"
    echo ""
    echo "Logs:"
    echo "  - logs/*.log"

else
    echo -e "\n${RED}✗ Experiments failed (exit code: $EXIT_CODE)${NC}"
    echo "Check logs: logs/experiment_${MODE}_${TIMESTAMP}.log"
    exit $EXIT_CODE
fi

echo -e "\n================================================================================"
echo -e "${GREEN}Done!${NC}"
echo "================================================================================"