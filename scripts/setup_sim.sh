#!/usr/bin/env bash
#
# Symlink Suturebot's XML configs, world URDFs, robot URDFs/meshes into
# the OpenSai config_folder so OpenSai_main can find them.
#
# Defaults to OpenSai at ../OpenSai (sibling of this repo). Override with:
#   OPENSAI=/path/to/OpenSai ./scripts/setup_sim.sh
#
# Idempotent: re-running just refreshes the symlinks.

set -euo pipefail

# Resolve repo root (parent of this script's dir)
SUTUREBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default OpenSai location: sibling directory
OPENSAI_DEFAULT="$(cd "$SUTUREBOT_ROOT/.." && pwd)/OpenSai"
OPENSAI="${OPENSAI:-$OPENSAI_DEFAULT}"

if [ ! -d "$OPENSAI/config_folder/xml_config_files" ]; then
    echo "Error: OpenSai not found at $OPENSAI" >&2
    echo "       (Expected $OPENSAI/config_folder/xml_config_files/ to exist.)" >&2
    echo "Set OPENSAI=/path/to/your/OpenSai to override." >&2
    exit 1
fi

echo "Suturebot: $SUTUREBOT_ROOT"
echo "OpenSai:   $OPENSAI"
echo

# --- Apply the OpenSai grasp patch (object pose/kinematic commands over redis) --
# Adds commands::<obj>::pose + ::kinematic to the simviz interface so a client
# can pin a dynamic object to the gripper (grip) or release it (drop). Idempotent.
PATCH="$SUTUREBOT_ROOT/scripts/opensai_patch.diff"
SAI_IF="$OPENSAI/core/sai-interfaces"
if [ -f "$PATCH" ] && [ -d "$SAI_IF/.git" ]; then
    if git -C "$SAI_IF" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        echo "OpenSai grasp patch: already applied."
    elif git -C "$SAI_IF" apply --check "$PATCH" >/dev/null 2>&1; then
        git -C "$SAI_IF" apply "$PATCH"
        echo "OpenSai grasp patch: APPLIED. You must rebuild OpenSai:"
        echo "  (cd \"$SAI_IF/build\" && make -j4) && (cd \"$OPENSAI/build\" && make -j4)"
    else
        echo "WARNING: opensai_patch.diff does not apply cleanly to $SAI_IF; skipping." >&2
    fi
else
    echo "Note: skipping grasp patch (no patch file or $SAI_IF is not a git repo)."
fi
echo

link_into() {
    local src="$1"
    local dst_dir="$2"
    local name
    name="$(basename "$src")"
    ln -sfn "$src" "$dst_dir/$name"
    echo "  $dst_dir/$name -> $src"
}

echo "Linking XML configs:"
for f in "$SUTUREBOT_ROOT"/config_folder/xml_config_files/*.xml; do
    [ -e "$f" ] || continue
    link_into "$f" "$OPENSAI/config_folder/xml_config_files"
done

echo
echo "Linking world URDFs:"
for f in "$SUTUREBOT_ROOT"/config_folder/world_files/*.urdf; do
    [ -e "$f" ] || continue
    link_into "$f" "$OPENSAI/config_folder/world_files"
done

echo
echo "Linking robot URDFs and standalone meshes:"
shopt -s nullglob
for f in "$SUTUREBOT_ROOT"/config_folder/robot_files/*.urdf \
         "$SUTUREBOT_ROOT"/config_folder/robot_files/*.stl \
         "$SUTUREBOT_ROOT"/config_folder/robot_files/*.obj; do
    link_into "$f" "$OPENSAI/config_folder/robot_files"
done
shopt -u nullglob

# Mesh tree (rizon4s/ — may itself be a symlink into Flexiv Files/).
if [ -e "$SUTUREBOT_ROOT/config_folder/robot_files/rizon4s" ]; then
    echo
    echo "Linking mesh tree:"
    link_into "$SUTUREBOT_ROOT/config_folder/robot_files/rizon4s" \
              "$OPENSAI/config_folder/robot_files"
fi

echo
echo "Done. Launch the sim with:"
echo "  cd \"$OPENSAI\""
echo "  sh scripts/launch.sh config_folder/xml_config_files/<your_xml>"
