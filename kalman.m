function [balltrack, speed] = kalman(videoname, minArea, SMALL_BALL)

% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();

tracks = initializeTracks(); % Create an empty array of tracks.
mask = [];
nextId = 1; % ID of the next track

% ballTrace = [];

% Detect moving objects, and track them across video frames.
frame_count = 1;
while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    
    displayTrackingResults();
    frame_count = frame_count+1;
end

% 寻找球的轨迹
tracknum = 0;
max_diff = 0;

for k = 1:3
   centerx = tracks(k).center(:,1);
   diff = max(centerx) - min(centerx);
   if diff > max_diff 
       max_diff = diff;
       tracknum = k;
   end
end


balltrack = tracks(tracknum);

% 去掉前十帧
balltrack.start = balltrack.start + 10;
centerTrace = balltrack.center;
centerTrace = double(centerTrace);
centerTrace = centerTrace(10:end, :);

% 计算整条 trace 中，两个连续点之间的平均距离
temp1 = centerTrace(2:end, 1) - centerTrace(1:end-1, 1);
temp2 = centerTrace(2:end, 2) - centerTrace(1:end-1, 2);
totald = 0;
for i = 1:size(temp1, 1)-1
    totald = totald + sqrt(temp1(i) + temp2(i));
end
meand = totald/(size(temp1,1)-1);

% 对轨迹中间的离群点进行线性插值
for i = 2 : size(centerTrace, 1)
    d = sqrt(  sum(  (centerTrace(i,:) - centerTrace(i-1, :)).^2  )  );
    if abs(d - meand)/meand > 2
        for j = i:min(i+10, size(centerTrace, 1))
            d1 = sqrt(  sum(  (centerTrace(j,:) - centerTrace(i-1, :)).^2  )  ) / (j-i+1);
            if abs(d1 - meand)/meand <= 2
                centerTrace(i-1:j, 1) = linspace(centerTrace(i-1,1), centerTrace(j,1), j-i+2);
                centerTrace(i-1:j, 2) = linspace(centerTrace(i-1,2), centerTrace(j,2), j-i+2);
                break
            end
        end
        if j == min(i+10, size(centerTrace, 1))
            centerTrace = centerTrace(1:i-1, :);
            break
        end
    end
end

ker = ones(1, 5)/5;
temp1 = centerTrace(:, 2);
% 地滚球不进行均值滤波（否则会导致大片水平线，导致计算出的速度显著下降）
if max(temp1)-min(temp1) > 20
    centerTrace = [conv(centerTrace(:, 1), ker) conv(centerTrace(:, 2), ker)];
    centerTrace = centerTrace(6:end-5, :);
end

balltrack.center = round(centerTrace);
d = balltrack.diameter;
BALLDIAMETER = median(d);
[speed_mean, speed_max, speed] = calspeed(balltrack.center);
fprintf('Average speed: %fm/s\n', speed_mean);
fprintf('Maximum speed: %fm/s\n', speed_max);


%% Create System Objects
% Create System objects used for reading the video frames, detecting
% foreground objects, and displaying results.

    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.
        
        % Create a video file reader.
        obj.reader = vision.VideoFileReader(videoname);
        
        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        % Create System objects for foreground detection and blob analysis
        
        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background. 
        
        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 60, 'MinimumBackgroundRatio', 0.7);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
%         
%         obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
%             'AreaOutputPort', true, 'CentroidOutputPort', true, ...
%             'MinimumBlobArea', 20);
%         
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', minArea); %'MaximumCount', 2);
    end

%% Initialize Tracks

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'start', {}, ...
            'diameter', {}, ...
            'bbox', {}, ...
            'center', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end

%% Read a Video Frame
% Read the next video frame from the video file.
    function frame = readFrame()
        frame = obj.reader.step();
    end

%% Detect Objects

    function [centroids, bboxes, mask] = detectObjects(frame)
        
        % Detect foreground.
        mask = obj.detector.step(frame);
        
        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15])); 
        mask = imfill(mask, 'holes');
        
        
        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            
            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            % Shift the bounding box so that its center is at 
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
            
            %我加的
            tracks(i).center = [tracks(i).center; predictedCentroid];
        end
    end

%% Assign Detections to Tracks

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 20;        
        
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

%% Update Assigned Tracks

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;
            
            %我加的
            tracks(trackIdx).center(end, :) = bbox(1:2) + 1/2*bbox(3:4);
            
            [~,~,y_min,y_max] = calbbox(mask(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3)));
            % mask有可能不准，但最后求直径会去掉离群值
            tracks(trackIdx).diameter = [tracks(trackIdx).diameter, y_max - y_min + 1];
            
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall. 

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 20;
        ageThreshold = 8;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];

        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = ((ages < ageThreshold & visibility < 0.6 ) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong);
        
        % Delete lost tracks.
        if ~SMALL_BALL
            tracks = tracks(~lostInds);
        end
    end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'start', frame_count, ...
                'diameter', [], ...
                'bbox', bbox, ...
                'center', centroid, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID 
% for each track on the video frame and the foreground mask. It then 
% displays the frame and the mask in their respective video players. 

    function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        
        minVisibleCount = 8;
        if ~isempty(tracks)
              
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                
                % Get ids.
                ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                
                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end
        
        % Display the mask and the frame.
        obj.maskPlayer.step(mask);        
        obj.videoPlayer.step(frame);
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

function [speed_mean, speed_max, speed] = calspeed(lines)
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
