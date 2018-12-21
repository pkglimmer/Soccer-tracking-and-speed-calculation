function kalmanShow(videoname, track)

obj = VideoReader(videoname);
firstframe = readFrame(obj);
figure,imshow(firstframe);

% 记录下初始坐标并存入lines中
count = 1;
lines = [];

frame = readFrame(obj);

start = track.start;
trace1 = double(track.center);
a = medfilt1(trace1(:, 1));
b = medfilt1(trace1(:, 2));
trace1 = [a, b];

k = 1;

while obj.CurrentTime<obj.Duration && count<size(trace1, 1)+start
%   对读入的每帧进行处理，结合上一帧的足球坐标（x,y），得到足球的分割前景图
imshow(frame)

%   记录下坐标存入lines中，并画图
hold on
if count >= start - 1 && k<=size(trace1, 1)
    lines(k,:) = trace1(k, :);
    plot(lines(:,1),lines(:,2),'r-');
    k = k + 1;
end

if k > size(trace1, 1)
    break
end

count = count+1;
hold off
%   (这里只是实例，你可能需要更好地存储方式)
%   滚动处理下一帧，直到结束


firstframe = frame;
frame = readFrame(obj);
end
end