# Soccer-tracking-and-speed-calculation
射门视频足球追踪与球速计算。基本方法须手动点击初始化足球位置；卡尔曼滤波可以自动识别足球。

Tracking.m and kalman.m provide the same functionality. 

In tracking.m, you have to manually mark the soccer's location in first frame of the video.
Kalman.m is based on the [matlab implementation of Kalman filter](https://ww2.mathworks.cn/help/vision/examples/motion-based-multiple-object-tracking.html), which automatically extracts 
the ball by simple condition judgements.

Note that the functions are all tested on iphone slow motion video clips, which means the speed calculation part are based on 240fps of frame rate.

For implementation details, see the .pdf project report.

**Parameters**:

* `tracking(videoname, KICK_RIGHT, SMALL_BALL)`
  - `videoname`: directory of the soccer (penalty kicking) video.
  - `KICK_RIGHT`: logical. 1 if the ball goes right, vice versa.
  - `SMALL_BALL`: logical. The diameter of a standard-sized soccer is 22cm, which corresponds to `SMALL_BALL=0`; if it's a toy ball (19.5 cm), assign SMALL_BALL to 0.
  
* `kalman(videoname, minArea, SMALL_BALL)`
  - `minArea`: a parameter for matlab function `configurekalmanfilter`, need to be adjusted according to soccer size in the video clip. (see demo.m)
 
* `kalmanShow`: visualization using soccer track (a struct returned by kalman.m script) and the input video.
