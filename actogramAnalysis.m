function [rawData, summaryStats, flyHMMs] = actogramAnalysis(varargin)
% ACTOGRAMANALYSIS  Makes an actogram out of text files and returns Hidden
% Markov Model parameters describing each individual fly's sleep patterns.
%
%    [RAWDATA, SUMMARYSTATS, FLYHMMS] = ACTOGRAMANALYSIS opens a GUI where
%    you select your .txt files and returns the raw numerical activity data
%    as RAWDATA and the data summary statistics as SUMMARYSTATS. The HMM
%    for each fly is returned as FLYHMMS.
%
%    [RAWDATA, SUMMARYSTATS, FLYHMMS] = ACTOGRAMANALYSIS('actogram') opens
%    a GUI where you select your .txt files and returns the raw numerical
%    activity data as RAWDATA and the data summary statistics as
%    SUMMARYSTATS and spits out an actogram.
%
%    [RAWDATA, SUMMARYSTATS, FLYHMMS] =
%    ACTOGRAMANALYSIS('PATH1','PATH2,...) opens .txt files at PATH1, PATH2,
%    etc. and returns the raw numerical actogram data as RAWDATA and the
%    data summary statistics as SUMMARYSTATS. If the first argument is
%    'actogram' (case-insensitive), it will also spit out actograms.
%
%    This is a program which takes in .txt files from the activity monitors
%    this lab seems to love so much, does some data processing, and creates
%    actograms from them. The format is fairly consistent across the
%    actograms, so if the script doesn't work, make sure there haven't been
%    any major changes.
%
%    Some of the summary statistics about bout lengths still need have
%    kinks that need to be worked out. I'll get to it soon (08/13/2014)
%
%    If the argument field is left blank, this will bring you to the GUI so
%    you can click the file(s) you want. Otherwise, type the path of the
%    files you want to analyze here.
%
% - SCT 07/18/2014

%% Importing the data

% Default setting: no actograms.
actogram = 0;

% First check to see if the varargin field is blank

if(nargin == 0)
    [filenames,pathnames] = uigetfile('*.txt', 'Multiselect','on');
    if(ischar(filenames))
        dataIn = importdata([pathnames,filenames]);
    else
        for j = 1:length(filenames)
            dataIn(j) = importdata([pathnames,filenames{j}]);
        end
    end
    
    % If it isn't, import the data specified on the command line
elseif(strcmpi(varargin{1},'actogram'))
    actogram = 1;
    if(nargin == 1)
        [filenames,pathnames] = uigetfile('*.txt', 'Multiselect','on');
        if(ischar(filenames))
            dataIn = importdata([pathnames,filenames]);
        else
            for j = 1:length(filenames)
                dataIn(j) = importdata([pathnames,filenames{j}]);
            end
        end
    end
    for j = 2:nargin
        dataIn(j-1) = importdata(varargin(j));
    end
else
    for j = 1:nargin
        dataIn(j) = importdata(varargin(j));
    end
end

% Note that, for the actograms readout I was given, the format is as
% follows:

% filename \t date numbins duration of bin (minutes) time of start Bin #1
% Bin #2 etc.

% First we extract all the metadata
years = [];
years = [years, dataIn(:).data];
txtdata = [];
txtdata = [txtdata,dataIn(:).textdata];
dates = [txtdata(1,2:3:end); txtdata(1,3:3:end)];

% Now we turn the numerical data into something we can actually use.

numBins = str2double(txtdata(2,1:3:end));
durs = str2double(txtdata(3,1:3:end));
% Note: startTimesCells is still a bunch of cells with strings
startTimesCells = txtdata(4,1:3:end);
% here are the beam crossings:
rawData = str2double(txtdata(5:end,1:3:end));
clear txtdata
%% Analyze data

% With the metadata out of the way, we can start actually looking at the
% ticks in a more interesting way than the actogram alone presents. Let's
% start with an HMM for sleep in which there are two states: a "deep" stage
% and a "shallow" or "restless" stage. We do this for each fly separately.
%
% excluding transitions back to itself:
%
%    Deep  <---->  Fitful  <----> Awake
%
% We have to separate on a fly-by-fly basis because they all will have a
% different total number of sleep bins.
%
% Note: in this section of the code, tasks are written mostly for clarity
% and not so much for speed. If this turns out to be a bit of a beast, I'll
% speed it up later. (07/26/14)

summaryStats = struct();
numFlies = size(rawData,2);
boutBins = cell([numFlies,1]);
timeAsleepPerDay = zeros(numFlies,1);
numDays = floor((numBins.*durs)/(60*24));
meanActivity = zeros(numFlies,1);
% Find sleep bouts under the "period of quiescence" definition
for j = 1:numFlies
    % Get data for this fly
    flyNumData = rawData(:,j);
    % Find candidate sleep bins
    candSleep = find(flyNumData == 0);
    % Minimum number of minutes a bout lasts (you define this)
    boutDuration = 5;
    boutBins{j} = candSleep(find(candSleep(1+boutDuration:end)-...
        candSleep(1:end-boutDuration) == boutDuration));
    timeAsleepPerDay(j) = length(boutBins{j})*durs(j)/numDays(j);
    meanActivity(j) = mean(flyNumData(flyNumData > 0));
end

summaryStats.averageTimeSleeping = mean(timeAsleepPerDay);
summaryStats.sleepTimeDistribution = timeAsleepPerDay;
clear timeAsleepPerDay;

% Keeping track of individual bouts (note: done on the flight to Anchorage
% while super sleepy. If there's a problem, look here)

individualBoutLengths = cell(numFlies, 1);
individualWakeBouts = cell(numFlies,1);
meanSleepBouts = zeros(numFlies,1);
meanWakeBouts = zeros(numFlies,1);

for j = 1:numFlies
    thisFlyBouts = boutBins{j};
    % For pesky tubes with dead flies
    if(isempty(find(rawData(:,j),1)))
    else
        % Start by looking for where there are abrupt changes in the fly
        % bout indices (i.e. going from one bout to the next)
        transitionIndices = find(thisFlyBouts(2:end)-thisFlyBouts(1:end-1) > 1);
        % This array actually gave us indices which are right before the
        % jump (so they're the ends of bouts)
        if(transitionIndices(1) == 1)
            boutStarts = [transitionIndices(1); thisFlyBouts(transitionIndices(2:end)-1)];
        else
            boutStarts = [thisFlyBouts(1); thisFlyBouts(transitionIndices(1:end-1)-1)];
        end
        boutEnds = thisFlyBouts(transitionIndices);
        if(boutEnds(end) < boutStarts(end))
            boutEnds = [boutEnds; thisFlyBouts(end)];
        end
        % FIXME: THIS PART IS BAD, I NEED TO FIGURE IT OUT
        individualBoutLengths{j} = boutEnds-boutStarts;
        individualWakeBouts{j} = boutStarts(2:end)-boutEnds(1:end-1);
        meanSleepBouts(j) = mean(individualBoutLengths{j});
        meanWakeBouts(j) = mean(individualWakeBouts{j});
    end
end

summaryStats.BoutLengths = individualBoutLengths;
summaryStats.WakeBouts = individualWakeBouts;
summaryStats.meanSleepBouts = meanSleepBouts;
summaryStats.meanWakeBouts = meanWakeBouts;
summaryStats.meanActivity = meanActivity;

% Build an HMM for activity readouts for each fly (sure to be trouble, will
% debug as I go along. -SCT 07/26/2014).
flyHMMs = cell(numFlies,4);
for j = 1:numFlies
    disp(['Analyzing fly number ', num2str(j)]);
    awakeState = struct();
    deepState = struct();
    fitfulState = struct();
    maxCrosses = max(rawData(rawData(:,j)>0,j));
    
    % Initialize descriptors with our priors. This part is a little tricky:
    % we might reasonably expect the duration of one continuous phase to be
    % its 1/its-self-transition-probability minutes. But obviously not all sleep or wake bouts are created equal.
    awakeState.transitionProbs = [1/meanWakeBouts(j), 1 - 1/meanWakeBouts(j), 0];
    fitfulState.transitionProbs = [1 - 1/meanSleepBouts(j), 1/(2*meanSleepBouts(j)), 1/(2*meanSleepBouts(j))];
    deepState.transitionProbs = [0, .05, .95];
    
    % We model the activity output for the awake state as a Poisson process
    % (this assumes that the activity when awake at night is the same as
    % the activity when awake in the day, which is not a good assumption,
    % but I'll see how it goes). The only descriptor necessary for this
    % probability density function is "lambda", so that the probability of
    % observing x bouts is:
    %
    % p(x) = e^(-lambda)*(lambda^x)/x!
    %
    % As our prior, we assume that the fly is only awake during the bins
    % in which there is a nonzero level of activity. We also assume that
    % sleeping flies do not have any activity. If X~Poiss(lambda), E[X] =
    % lambda.
    
    awakeState.lambda = meanActivity(j);
    
    awakeState.emission = exp(-awakeState.lambda)*awakeState.lambda.^...
        [0:1:maxCrosses]./factorial(0:1:maxCrosses);
    
    fitfulState.emission = [1,0*[1:maxCrosses]];
    
    deepState.emission = fitfulState.emission;
    
    transGuess = [awakeState.transitionProbs; fitfulState.transitionProbs;...
        deepState.transitionProbs];
    
    emisGuess =[awakeState.emission; fitfulState.emission; deepState.emission];
    
    % +1 because hmmtrain only works for sequences of ints >= 1
    
    [tranEst, emisEst] = hmmtrain(rawData(:,j)+1,transGuess,emisGuess,...
        'VERBOSE', true);
    
    awakeState.transitionProbs = tranEst(1,:);
    awakeState.emission = emisEst(1,:);
    fitfulState.transitionProbs = tranEst(2,:);
    fitfulState.emission = emisEst(2,:);
    deepState.transitionProbs = tranEst(3,:);
    deepState.emission = emisEst(3,:);
    
    flyHMMs{j,1} = awakeState;
    flyHMMs{j,2} = fitfulState;
    flyHMMs{j,3} = deepState;
    % FIXME: DECODE SYMBOL SET PROBLEM ? ? ? ? ? ?
    flyHMMs{j,4} = hmmdecode(rawData(:,j)+1,tranEst,emisEst);
end


%% Create an actogram
if (actogram == true)
    for i = 1:numFlies
        % A new figure for every fly
        figure;
        % A fix for the whole 'one file means the names aren't a cell'
        % thing
        if(numFlies == 1)
            suptitle([filenames,char(10), dates{1,i}, ' ', dates{2,i},' ', ...
                num2str(years(i)),char(10)]);
        else
            suptitle([filenames{i},char(10), dates{1,i}, ' ', dates{2,i},' ', ...
                num2str(years(i)),char(10)]);
        end
        boutTimes = boutBins{i};
        % bpd = bins per day
        bpd = 1440 / durs(i);
        for k = 1:numDays(i)
            % And a new subplot for every day
            subplot(numDays(i),1,k)
            todaysData = rawData(1+(bpd*(k-1)):bpd*k,i);
            % Now we plot the actual actogram
            ticks = (1:bpd)*durs(i);
            % This is for making a blue background for bouts and a red
            % background during non-sleep
            todaysBouts = boutTimes(1+(bpd*(k-1)) <= boutTimes & ...
                boutTimes <= bpd*k) - (bpd*(k-1));
            maxNum = max(todaysData);
            awakeTimes = todaysData > 0;
            wake = bar(ticks,(maxNum+1)*awakeTimes,...
                'FaceColor', [238/255,121/255,159/255],'EdgeColor','none','BarWidth',1);
            hold on;
            bts = bar(todaysBouts, (maxNum+1)*ones(length(todaysBouts),1), ...
                'FaceColor', [144/255,230/255,230/255], ...
                'EdgeColor', 'none', 'BarWidth', 1);
            % plotting the actual data
            foreg = bar(ticks,todaysData,'k');
            xlim([0,1440]);
            ylim([0,maxNum+1]);
            % boring formatting stuff
            set(gca,'TickDir','out');
            hold off;
            %pcg = get(bkg,'child'); % get patch objects from barseries
            %object set(pcg,'FaceAlpha',.4); % set transparencies
        end
        xlabel('Minutes into day');
    end
end



