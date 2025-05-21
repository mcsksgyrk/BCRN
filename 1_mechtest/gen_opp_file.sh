#!/bin/bash
# Examples:
#   ./generate_mechmod.sh my_config.txt 1000 my_folder   # Custom folder

# Default values
OUTPUT_FILE="${1:-MECHMOD_expanded.txt}"
END_NUM="${2:-1000}"
FOLDER_NAME="${3:-20240831_stac_8adsigma}"
PREFIX="stac_"
SUFFIX=".xml"
BASE_PATH="7_Krisztian/xml"
PADDING=4  # Number of digits to pad with zeros (e.g., 0001)

# Create the header section
cat > "$OUTPUT_FILE" << EOF
MECHMOD
  USE_NAME         BCRN6
  MECH_FILE        7_Krisztian/mech/BCRN6.inp
  COMPILE_cantera  7_Krisztian/mech/BCRN6.yaml
END
MECHTEST
    MECHANISM  BCRN6
    TIME_LIMIT 50
    THREAD_LIMIT 32
    SETTINGS_TAG systems_biology
    FALLBACK_TO_DEFAULT_SETTINGS
    
    SOLVER cantera
    SAVE_STATES      CSV
EOF

# Generate the file lines with proper padding
for ((i=1; i<=END_NUM; i++)); do
    # Format the number with leading zeros based on PADDING
    NUM=$(printf "%0${PADDING}d" $i)
    echo "      NAME $BASE_PATH/$FOLDER_NAME/$PREFIX$NUM$SUFFIX" >> "$OUTPUT_FILE"
done

# Add the closing END
echo "END" >> "$OUTPUT_FILE"

echo "Generated $OUTPUT_FILE with $END_NUM entries from $PREFIX$(printf "%0${PADDING}d" 1)$SUFFIX to $PREFIX$(printf "%0${PADDING}d" $END_NUM)$SUFFIX in folder $BASE_PATH/$FOLDER_NAME/"