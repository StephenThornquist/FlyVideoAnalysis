function zhangFlyTracker(nfly,videopath,vargin)
% ZHANGFLYTRACKER  Track fruit flies in a video
%    ZHANGFLYTRACKER(NFLY) lets you select the video(s) using a GUI.
%    ZHANGFLYTRACKER(NFLY,VIDEOPATH) tracks the movie at VIDEOPATH in which there are
%    NFLY flies. (VIDEOPATH can be a cell array of strings of their paths).
%    ZHANGFLYTRACKER(NFLY,VIDEOPATH,1) puts the script in "quiet mode" so
%    that it doesn't display output to your MATLAB command window. By
%    default, this is set to "off".
%
%    Stephen Zhang's fly tracking script. I don't know when it was made or
%    what much of the functionality is. I'll try to make sense of it and
%    introduce my own comments and tweaks if I have time/interest/need etc.
%    Those comments will have my name attached, so if something suddenly
%    doesn't work, look to see whose name is on the comments before you
%    decide who to ask.
%
% -SCT 07/21/2014

%% LOAD INPUTS
% Find out what videos we want to analyze -SCT 07/22/2014
if(nargin == 1)
        [filenames,pathnames] = uigetfile('*.MP4', 'Multiselect','on');
        if(ischar(filenames))
            video = VideoReader([pathnames,filenames]);
        else
            % Unfortunately, the way VideoReader is structured makes me
            % unable to preallocate memory here. Since videos are pretty
            % big, reallocating the arrays every time it adds a video is
            % going to eat up a LOT of memory. Hopefully that'll be okay;
            % if not, I'll find a fix. -SCT 07/22/2014
            for j = 1:length(filenames)
                video(j) = VideoReader([pathnames,filenames{j}]);
            end
        end
        quietmode = 0;
else
    for j = 1:length(videopath)
        video(j) = VideoReader(videopath{j});
    end
    
    if(nargin == 2)
        quietmode = 0;
    else
        quietmode = vargin{1};
    end
end

%% TRACK FLIES

% In this cell we actually track the flies.

% Okay so here a for loop is okay, since the overhead is practically
% trivial compared to the contents of each loop element. What we're doing
% is looping over every video in the list of videos passed in and tracking
% each of the flies. -SCT 07/22/2014

for k = 1:length(video)
    currVid = video(k);
    nframe = currVid.NumberOfFrames;
    
    clear props; % This step seems needed no matter what
    props(1:nfly,1:nframe)=struct('Area',[],'Centroid',zeros(1,2,'double'),...
        'MajorAxisLength',[],'MinorAxisLength',[],'Eccentricity',[],...
        'Orientation',[]);
    if quietmode==0
        dispbar=waitbar(0,['Tracking Video',currVid.Name,...
            '-Arena',num2str(arena_num)]);
    end

    for i=1:nframe
        arena_rev_nbg=Arena(:,:,i)-background{arena_num};
        arena_rev_nbg=imadjust(arena_rev_nbg,[0 0.5],[0 1],gamma);
        %Re-adjust gamma. Numbers can be optimized.
        arena_rev_nbg_bw=im2bw(arena_rev_nbg,...
            custom_bw_threshold_modifier*std(reshape(mat2gray(arena_rev_nbg),...
            1,[]))+mean(reshape(mat2gray(arena_rev_nbg),1,[])));

        % remove the moon-shaped ring before erosion
        [tobedemooned,n2testdemooned]=bwlabel(arena_rev_nbg_bw);
        tobedemooned_struct=regionprops(tobedemooned,'Extrema');
        demoon_index=[];
        for shade_ind=1:n2testdemooned
            shade_size=range(tobedemooned_struct(shade_ind).Extrema);
            if shade_size(1)*shade_size(2)>demoon_cutoff
                demoon_index=[demoon_index;shade_ind];
                Flags_demooned(i,arena_num)=shade_size(1)*shade_size(2);
            end
        end
        for shade_ind=1:size(demoon_index)
            tobedemooned(tobedemooned==demoon_index(shade_ind))=0;
        end
        arena_rev_nbg_bw=tobedemooned>0;

        arena_rev_nbg_bw_erode(:,:,i)=imerode(arena_rev_nbg_bw,strel('disk',flysize));
        [arena_rev_nbg_bw_erode_lb(:,:,i),nflydetected(i)]=bwlabel(arena_rev_nbg_bw_erode(:,:,i));

        if nflydetected(i)<nfly % Need watershed
            %disp(['Frame ' , num2str(i) , ' Watershedding'])
            n_watershed=n_watershed+1;
            %fwatershed=[fwatershed;i];
            tobewatershed=arena_rev_nbg_bw_erode(:,:,i);
            shedbound=watershed(-bwdist(~tobewatershed));
            tobewatershed(shedbound==0)=0;
            [arena_rev_nbg_bw_erode_lb(:,:,i),nflydetected(i)]=bwlabel(tobewatershed);

            if nflydetected(i)>nfly % Anti-overwatershed
                n_anti_overshed=n_anti_overshed+1;
                %fanti_overshed=[fanti_overshed;i];
                tobeantiovershed=arena_rev_nbg_bw_erode_lb(:,:,i); %
                %Automatic anti-overshedding sounds pretty difficult, so for now, I will
                %choose the two deepest sinks as fly approximations.
                tobewatershed(tobeantiovershed>2)=0;
                [arena_rev_nbg_bw_erode_lb(:,:,i),nflydetected(i)]=bwlabel(tobewatershed);
                Flags(i,arena_num)=3;
            end

            if nflydetected(i)<nfly % Need force segmentation
                %disp('Watershedding Unsuccessful')
                %disp(['Frame ' , num2str(i) , ' Forced Segmenting'])
                n_force_seg=n_force_seg+1;
                %fforce_seg=[fforce_seg;i];
                tobeforcesegmented=arena_rev_nbg_bw_erode(:,:,i); %
                %Initiate dot removal
                tobeforcesegmented_dist=bwdist(~tobeforcesegmented);
                force_zero_cand_index=find(tobeforcesegmented_dist==1);
                force_segmented_lb=ones([size(tobeforcesegmented),length(force_zero_cand_index)]).*NaN;
                force_segmented_nfly_det=zeros(length(force_zero_cand_index),1).*NaN;

                for j=1:length(force_zero_cand_index)
                    force_segment_try=tobeforcesegmented;
                    force_segment_try(force_zero_cand_index(j))=0;
                    [force_segmented_lb(:,:,j),force_segmented_nfly_det(j)]=bwlabel(force_segment_try);
                end

                succ_force_seg_index=find(force_segmented_nfly_det==nfly); % This is not
                %the pixel index

                if succ_force_seg_index>0
                    Flags(i,arena_num)=4;
                    %disp('Dot removal Successful')
                    succ_force_segmented_lb=force_segmented_lb(:,:,succ_force_seg_index);
                    succ_force_segmented_nfly_det=force_segmented_nfly_det(succ_force_seg_index);
                    succ_force_zero_index=force_zero_cand_index(succ_force_seg_index);

                    default_force_seg_choice=1; % Set the first successful
                    one to be default

                    arena_rev_nbg_bw_erode_lb(:,:,i)=succ_force_segmented_lb(:,:,default_force_seg_choice);
                    % Choose the first successful one
                    nflydetected(i)=succ_force_segmented_nfly_det(default_force_seg_choice);
                    forcesegmented=tobeforcesegmented;
                    forcesegmented(succ_force_zero_index(default_force_seg_choice))=0;
                    arena_rev_nbg_bw_erode(:,:,i)=forcesegmented;
                else
                    %disp('Dot removal Unsuccessful')
                    tobeexringed=arena_rev_nbg_bw_erode(:,:,i); % Initiate
                    %external ring removal
                    tobeexringed(tobeforcesegmented_dist>=1 & ...
                    tobeforcesegmented_dist<2)=0;
                    [arena_rev_nbg_bw_erode_lb(:,:,i),nflydetected(i)]=bwlabel(tobeexringed);
                    if nflydetected(i)==nfly
                        %disp('External Ring Removal Successful')
                        Flags(i,arena_num)=5;
                        arena_rev_nbg_bw_erode(:,:,i)=tobeexringed;
                    else
                        %disp('External Ring Removal Unsuccessful') %
                        %Initiate internal ring removal
                        tobeinringed=arena_rev_nbg_bw_erode(:,:,i);
                        tobeinringed(tobeforcesegmented_dist>=2 & ...
                        tobeforcesegmented_dist<=3)=0;
                        [arena_rev_nbg_bw_erode_lb(:,:,i),nflydetected(i)]=bwlabel(tobeinringed);
                        if nflydetected(i)==nfly
                            %disp('Internal Ring Removal Successful')
                            Flags(i,arena_num)=6;
                        else
                            %disp('Internal Ring Removal Unsuccessful') %
                            %In this very extreme case, the two flies are essentially on top of each
                            %other
                            %disp('Fly Created')
                            Flags(i,arena_num)=7;
                            n_created=n_created+1;
                            %fcreate=[fcreate;i];
                            tocreatefly=arena_rev_nbg_bw_erode(:,:,i); % I
                            %have only considered the 2 fly situation here.
                            if sum(sum(tocreatefly))==0
                                if i>1
                                    tocreatefly=arena_rev_nbg_bw_erode(:,:,i-1); % If This frame has no
                                    %pixel, use the last frame (temporary solution)
                                else
                                    tocreatefly=arena_rev_nbg_bw;
                                end
                            end
                            tocreatefly_index=find(tocreatefly==1);
                            createfly=tocreatefly*0;
                            fly_creation_random_number=randi(sum(sum(tocreatefly)));
                            createfly(tocreatefly_index(fly_creation_random_number))=1;
                            createfly(tocreatefly_index(fly_creation_random_number)+round(randi(2)-1.5))=2;
                            % Fly 2 above/below 1 pixel of Fly 1
                            arena_rev_nbg_bw_erode_lb(:,:,i)=createfly;
                            nflydetected(i)=2;
                            arena_rev_nbg_bw_erode(:,:,i)=createfly>0;
                        end
                    end
                end

            else
                %disp('Watershedding Successful')
                Flags(i,arena_num)=2;
                arena_rev_nbg_bw_erode(:,:,i)=tobewatershed; % Write the
                %watershed result to the eroded image if no force segmentation
            end

        elseif nflydetected(i)>nfly % Reduce arena if too many flies
            n_reduce=n_reduce+1;
            %freduce=[freduce;i];
            %disp(['Frame ' , num2str(i) , ' fly number reduced'])
            tobereducedarena=arena_rev_nbg_bw_erode_lb(:,:,i);
            tobedeletedarea_struct=regionprops(arena_rev_nbg_bw_erode_lb(:,:,i),'Area');
            tobedeletedarea=[tobedeletedarea_struct.Area]';
            areastokeep=sort(tobedeletedarea,'descend');
            areatokeep=areastokeep(1:nfly);
            areatokeepindex=zeros(nfly,1);

            for j=1:nfly
                areatokeepindex(j)=find(tobedeletedarea==areatokeep(j),1); % Min just to
                %avoid error message
                tobereducedarena(tobereducedarena==areatokeepindex(j))=25000;
            end

            reducedarena=tobereducedarena;
            reducedarena(reducedarena<=24999)=0;
            reducedarena(reducedarena>24999)=1;

            [arena_rev_nbg_bw_erode_lb(:,:,i),nflydetected(i)]=bwlabel(reducedarena);
            arena_rev_nbg_bw_erode(:,:,i)=reducedarena;
            Flags(i,arena_num)=1;
        end

        props(:,i)=regionprops(arena_rev_nbg_bw_erode_lb(:,:,i),'Centroid', ...
            'Orientation','Area','Eccentricity','MajorAxisLength','MinorAxisLength');
        if quietmode==0
            waitbar(i/nframe,dispbar)
        end
    end
    arena_rev_nbg_bw_erode_lb=uint8(arena_rev_nbg_bw_erode_lb);
    arena_rev_nbg_bw_erode=uint8(arena_rev_nbg_bw_erode);
end