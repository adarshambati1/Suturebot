// RedisTeleopBridge.cs
//
// Drop into a Unity project that has:
//   - Meta XR Core SDK (for OVRInput / OVRCameraRig)
//   - StackExchange.Redis (via NuGetForUnity or DLL drop)
//
// Attach to an empty GameObject in the scene. Wire up the inspector fields.
//
// Writes match the Redis keys consumed by the existing controller.cpp
// stack, mirroring python_examples/suturebot_grav_motion.py:
//
//   opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position
//   opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation
//   opensai::controllers::Rizon4s::cartesian_controller::gripper_fingers::goal_position
//
// Position is JSON [x, y, z]. Orientation is JSON 3x3 row-major [[r11..r13],..].

using System;
using System.Globalization;
using System.Text;
using UnityEngine;
using StackExchange.Redis;

public enum ScenePreset
{
    // suturebot_grav_pierce_oussama_push.xml — hemostat gripper, +X pierce.
    OussamaPush,
    // suturebot_grav_motion.py — foam block, +Y pierce direction.
    GravMotion,
}

public class RedisTeleopBridge : MonoBehaviour
{
    [Header("Redis")]
    public string redisHost = "127.0.0.1";
    public int redisPort = 6379;

    [Header("Scene")]
    [Tooltip("Which OpenSai XML scene this bridge is driving. " +
             "Sets the home pose the robot returns to on deadman engage " +
             "and the fixed end-effector orientation.")]
    public ScenePreset scene = ScenePreset.OussamaPush;

    [Header("Scene anchors")]
    [Tooltip("RightControllerAnchor under your OVRCameraRig.")]
    public Transform rightControllerAnchor;

    [Tooltip("Empty GameObject placed at the Rizon4s base in Quest world. " +
             "Controller pose is expressed relative to this transform, then " +
             "remapped to the robot frame.")]
    public Transform robotBaseAnchor;

    [Header("Teleop tuning")]
    [Tooltip("Multiplier on hand displacement -> robot displacement. " +
             "Suturing is millimetric; keep this < 1.")]
    [Range(0.05f, 1.0f)] public float motionScale = 0.3f;

    [Tooltip("Pose updates per second written to Redis.")]
    [Range(10, 120)] public int updateRateHz = 60;

    [Tooltip("Only stream commands while the right grip button is held. " +
             "This is the deadman switch — release to freeze the robot.")]
    public bool useDeadman = true;

    [Header("Workspace clamp (robot base frame, meters)")]
    public Vector3 workspaceMin = new Vector3(0.30f, -0.30f, 0.00f);
    public Vector3 workspaceMax = new Vector3(0.70f,  0.30f, 0.40f);

    // Redis keys — keep in sync with python_examples/suturebot_grav_motion.py.
    const string K_GOAL_POS  = "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position";
    const string K_GOAL_ORI  = "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation";
    const string K_GRIP      = "opensai::controllers::Rizon4s::cartesian_controller::gripper_fingers::goal_position";

    ConnectionMultiplexer _redis;
    IDatabase _db;
    float _writePeriod;
    float _nextWriteTime;

    // Captured on (re)engage of the deadman so subsequent motion is a delta
    // from the moment the user pressed the grip button.
    Vector3 _engagePosRobot;
    Quaternion _engageOriRobot;
    Vector3 _engageControllerPos;
    Quaternion _engageControllerOri;
    bool _engaged;

    void Start()
    {
        if (rightControllerAnchor == null || robotBaseAnchor == null)
        {
            Debug.LogError("[RedisTeleopBridge] Assign rightControllerAnchor and robotBaseAnchor in the inspector.");
            enabled = false;
            return;
        }

        try
        {
            _redis = ConnectionMultiplexer.Connect($"{redisHost}:{redisPort}");
            _db = _redis.GetDatabase();
            Debug.Log($"[RedisTeleopBridge] Connected to Redis at {redisHost}:{redisPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[RedisTeleopBridge] Redis connect failed: {e.Message}");
            enabled = false;
            return;
        }

        _writePeriod = 1.0f / updateRateHz;
        _nextWriteTime = Time.time;

        // Seed the robot's "engaged" pose from the scene preset. (TODO: fetch
        // the robot's actual current pose from Redis on engage instead — this
        // hardcoded value snaps the EE the first time you press the deadman.)
        _engagePosRobot = HomeForScene(scene);
        _engageOriRobot = Quaternion.identity;  // unused; emitter writes a literal matrix
    }

    void Update()
    {
        if (_db == null) return;
        if (Time.time < _nextWriteTime) return;
        _nextWriteTime += _writePeriod;

        bool deadmanHeld = !useDeadman ||
            OVRInput.Get(OVRInput.Button.PrimaryHandTrigger, OVRInput.Controller.RTouch);

        if (!deadmanHeld)
        {
            _engaged = false;
            return;
        }

        // On the rising edge of the deadman, snapshot the controller and robot
        // poses. From here on we send (robot_engage + scale * (controller - controller_engage)).
        if (!_engaged)
        {
            _engageControllerPos = rightControllerAnchor.position;
            _engageControllerOri = rightControllerAnchor.rotation;
            // _engagePosRobot / _engageOriRobot stay at whatever the robot is
            // currently commanded to. (TODO: read back from Redis to stay in sync.)
            _engaged = true;
        }

        // Controller delta in Quest world.
        Vector3 dPosQuest = rightControllerAnchor.position - _engageControllerPos;

        // Express in robot base frame: parent the delta under robotBaseAnchor's
        // inverse rotation, then handle Unity-(left-handed Y-up) to robot-(right-handed Z-up)
        // axis swap. Easiest: do it explicitly.
        Vector3 dPosRobot = QuestToRobot(robotBaseAnchor.InverseTransformVector(dPosQuest));
        Vector3 goalPos = _engagePosRobot + motionScale * dPosRobot;
        goalPos = ClampToWorkspace(goalPos);

        // Orientation: hold it fixed at the scene's expected EE orientation.
        // Hooking up controller orientation cleanly takes one more calibration pass.
        _db.StringSet(K_GOAL_POS, Vec3ToJson(goalPos));
        _db.StringSet(K_GOAL_ORI, Mat3ToJson(OriMatrixForScene(scene)));

        // Trigger axis 0..1 -> gripper width. Closed = 5 mm, open = 50 mm.
        float trig = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
        float width = Mathf.Lerp(0.05f, 0.005f, trig);
        _db.StringSet(K_GRIP, $"[{width.ToString("F4", CultureInfo.InvariantCulture)}]");
    }

    void OnDestroy()
    {
        _redis?.Dispose();
    }

    // ---- helpers ----------------------------------------------------------

    // Unity is left-handed, Y-up. Robot is right-handed, Z-up.
    // Quest world  (x_right, y_up,   z_forward)
    // Robot base   (x_fwd,   y_left, z_up)
    // Map: robot.x = quest.z, robot.y = -quest.x, robot.z = quest.y
    static Vector3 QuestToRobot(Vector3 q) => new Vector3(q.z, -q.x, q.y);

    Vector3 ClampToWorkspace(Vector3 p) => new Vector3(
        Mathf.Clamp(p.x, workspaceMin.x, workspaceMax.x),
        Mathf.Clamp(p.y, workspaceMin.y, workspaceMax.y),
        Mathf.Clamp(p.z, workspaceMin.z, workspaceMax.z));

    static Vector3 HomeForScene(ScenePreset s) => s switch
    {
        // suturebot_grav_pierce_oussama_push.py: X_HOME=0.6686, Y first stitch=0.1485,
        // WORKING_Z=0.3146
        ScenePreset.OussamaPush => new Vector3(0.6686f, 0.1485f, 0.3146f),
        // suturebot_grav_motion.py: foam-block home
        ScenePreset.GravMotion  => new Vector3(0.51f, -0.20f, 0.03f),
        _ => Vector3.zero,
    };

    static float[,] OriMatrixForScene(ScenePreset s) => s switch
    {
        // suturebot_grav_pierce_oussama_push.py: ORI = [[1,0,0],[0,-1,0],[0,0,-1]]
        ScenePreset.OussamaPush => new float[,] {
            { 1f,  0f,  0f },
            { 0f, -1f,  0f },
            { 0f,  0f, -1f },
        },
        // suturebot_grav_motion.py: ORI_DOWN = [[0,-1,0],[-1,0,0],[0,0,-1]]
        ScenePreset.GravMotion => new float[,] {
            { 0f, -1f,  0f },
            {-1f,  0f,  0f },
            { 0f,  0f, -1f },
        },
        _ => new float[,] { {1,0,0},{0,1,0},{0,0,1} },
    };

    static string Vec3ToJson(Vector3 v) =>
        $"[{F(v.x)},{F(v.y)},{F(v.z)}]";

    static string Mat3ToJson(float[,] m)
    {
        var sb = new StringBuilder("[");
        for (int r = 0; r < 3; r++)
        {
            if (r > 0) sb.Append(",");
            sb.Append("[");
            for (int c = 0; c < 3; c++)
            {
                if (c > 0) sb.Append(",");
                sb.Append(F(m[r, c]));
            }
            sb.Append("]");
        }
        sb.Append("]");
        return sb.ToString();
    }

    static string F(float x) => x.ToString("F6", CultureInfo.InvariantCulture);
}
