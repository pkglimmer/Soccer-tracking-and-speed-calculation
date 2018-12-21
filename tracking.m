function [speed_mean, speed_max, speed] = tracking(videoname, KICK_RIGHT, SMALL_BALL)
% 后两个输入参数均为逻辑值：KICK_RIGHT = 1表示向右踢；SMALL_BALL = 1 表示球是小号的。
% gpk
% 2018.12.15

% xzw
% 2018.12.5

obj = VideoReader(videoname);
firstframe = readFrame(obj);
figure;imshow(firstframe);
title('点选球的中心点并回车')
[x,y] = initial();
x_last = x;
y_last = y;
%%
% 记录下初始坐标并存入lines中
count = 1;
lines = [];
radius = [];
lines(count, :) = [x,y];
%%
segs = [];
frame = readFrame(obj);
while obj.CurrentTime<obj.Duration
%   对读入的每帧进行处理，结合上一帧的足球坐标（x,y），得到足球的分割前景图
[ballground, x_last, y_last] = segment(frame, firstframe, x, y, x_last, y_last);
imshow(frame)
%   根据足球前景图确定足球的中心
[x,y] = calcenter(ballground, x, y, KICK_RIGHT);
%   记录下坐标存入lines中，并画图
%   记录下这一帧的足球前景图，并存入segs中
count = count+1;
hold on
lines(count,:) = [x,y];
plot(lines(:,1), lines(:,2),'r-');
title('Ball tracing');
hold off
[~,~,y_min,y_max] = calbbox(ballground);
%   (这里只是实例，你可能需要更好地存储方式)
segs(count-1,:,:) = ballground;
%   滚动处理下一帧，直到结束
for i = 1:3
    per_frame = frame(:,:,i);
    per_frame(ballground>0) = 255;
    frame(:,:,i)= per_frame;
end
radius(count) = y_max - y_min + 1;
firstframe = frame;
frame = readFrame(obj);
end
%%
% 处理完每一帧后，根据保存的足球分割图集segs，结合一些先验知识，计算足球面积、估算球速等
% 先一维均值滤波平滑再计算球速
ker = ones(1, 5)/5;
lines = [conv(lines(:, 1), ker) conv(lines(:, 2), ker)];
lines = lines(6:end-5, :);


lines = double(lines);
lines = lines(10:end, :);

% 计算整条 trace 中，两个连续点之间的平均距离
temp1 = lines(2:end, 1) - lines(1:end-1, 1);
temp2 = lines(2:end, 2) - lines(1:end-1, 2);
totald = 0;
for i = 1:size(temp1, 1)-1
    totald = totald + sqrt(temp1(i) + temp2(i));
end
meand = totald/(size(temp1,1)-1);
for i = 2 : size(lines, 1)
    d = sqrt(  sum(  (lines(i,:) - lines(i-1, :)).^2  )  );
    if abs(d - meand)/meand > 2
        for j = i:min(i+20, size(lines, 1))
            d1 = sqrt(  sum(  (lines(j,:) - lines(i-1, :)).^2  )  ) / (j-i+1);
            if abs(d1 - meand)/meand <= 2
                lines(i-1:j, 1) = linspace(lines(i-1,1), lines(j,1), j-i+2);
                lines(i-1:j, 2) = linspace(lines(i-1,2), lines(j,2), j-i+2);
                break
            end
        end
        if j == min(i+10, size(lines, 1))
            lines = lines(1:i-1, :);
            break
        end
    end
end 
lines = round(lines);




% 取半径采样结果的中位数作为半径
BALLDIAMETER = median(radius);
[speed_mean, speed_max, speed] = calspeed(segs);
fprintf('Average speed: %fm/s\n', speed_mean);
fprintf('Maximum speed: %fm/s\n', speed_max);

%% 平滑后效果作图
%   下面是最终展示足球框的代码
    figure,imshow(frame)
    hold on
    plot(lines(:,1), lines(:,2),'r-');
    [x_min,x_max,y_min,y_max] = calbbox(ballground);
    draw_lines(x_min,x_max,y_min,y_max)
    hold on
    title('平滑后轨迹');

%%
function draw_lines(x_min,x_max,y_min,y_max)
    hold on
    liness = [x_min,x_min,x_max,x_max,x_min;y_min,y_max,y_max,y_min,y_min];
    plot(liness(1,:),liness(2,:));
    hold off;
end

function [x_min,x_max,y_min,y_max] = calbbox(I)
    [rows,cols] = size(I); 
    temp1 = ones(rows,1)*[1:cols];
    temp2 = [1:rows]'*ones(1,cols);   
    rows = I.*temp1;
    x_max = max(rows(:))+2;
    rows(rows==0) = x_max;
    x_min = min(rows(:))-2;
    rows = I.*temp2;
    y_max = max(rows(:))+2;
    rows(rows==0) = y_max;
    y_min = min(rows(:))-2;
end
%%

function [x,y] = initial()
%         你需要在这里完成足球点的初始化
%         示例代码
        [x,y] = ginput();
        x = int16(x);
        y = int16(y);
end

function [ballground, x_last, y_last] = segment(frame,firstframe, x, y, x_last, y_last)
%         你需要在这里完成每一帧的足球前景分割
%         示例代码
    h = 20;
    w = 20;
    bw = rgb2gray(firstframe);
    bw2 = rgb2gray(frame);
    mask = zeros(size(bw2));
    % 把 ROI 按照惯性方向移动
    mask( 2*y-y_last-h: 2*y-y_last+h, 2*x-x_last-w:2*x-x_last+w ) = 1;
    % 在移动后的区域进行阈值化
    ballground = mask & (abs(bw2-bw)>20);
    x_last = x;
    y_last = y; % 记下当前位置，每次根据上次位置迭代
    B = ones(2); 
    ballground = imerode(ballground,B);
    ballground = imdilate(ballground,B);    
end

function [meanx,meany] = calcenter(I, x, y, KICK_RIGHT)
%         你需要在这里完成足球中心点的计算，根据前景图
%         示例代码
    [rows,cols] = size(I); 
    tempx = ones(rows,1)*[1:cols];
    tempy = [1:rows]'*ones(1,cols);   
    area = sum(sum(I)); 
    meanx = int16(sum(sum(I.*tempx))/area); 
    meany = int16(sum(sum(I.*tempy))/area);
    if((KICK_RIGHT && meanx<=x) || (~KICK_RIGHT && meanx>=x))
        meanx = x;
        meany = y;
    end 
end

function [speed_mean, speed_max, speed] = calspeed(segs)
%         你需要在这里完成足球面积的计算和球速的估算
%  先两个两个点计算，再按一定分位数舍弃离群值
        if SMALL_BALL
            D = 19;
        else
            D = 21.5;
        end
        speed = [];
        temp1 = lines(2:end, 1) - lines(1:end-1, 1);
        temp2 = lines(2:end, 2) - lines(1:end-1, 2);
        v = temp1.^2 + temp2.^2;
        v = v.^0.5;
        % 去掉前后一定的分位数计算平均速度
        v = v(  v > quantile(v, 0.05) & v < quantile(v, 0.95) );
        v = v / BALLDIAMETER * D / 100 * 240;
        figure;
        histogram(v);
        title(['Velocity: ' videoname]);
        speed_max = max(v);
        speed_mean = mean(v);
end
end


