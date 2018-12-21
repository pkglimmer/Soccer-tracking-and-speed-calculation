%%
% 第二个参数是向右踢，第三个参数"是小号球"则为1，是大球则为0
close all; tracking('./videos/7.mp4', 1, 0);


%%
% 第二个参数是小号球则为1，是大球则为0
close all;
videoname = './videos/0.mp4';
% 如果是第0个视频，第二个参数，最小检测区域，设为100；
[balltrack, speed] = kalman(videoname, 100, 1);

kalmanShow(videoname, balltrack)
title([videoname, '  Kalman+smoothed']);

%%
% 第二个参数是小号球则为1，是大球则为0
close all;
videoname = './videos/2.mp4';
% 如果不是第0个视频，第二个参数，最小检测区域，设为20；
[balltrack, speed] = kalman(videoname, 20, 0);

kalmanShow(videoname, balltrack)
title([videoname, '  Kalman+smoothed']);
