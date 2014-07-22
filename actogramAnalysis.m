function actogramAnalysis(vargin)
% ACTOGRAMANALYSIS  Makes an actogram out of text files
%    ACTOGRAMANALYSIS opens a GUI where you select your .txt files
%    ACTOGRAMANALYSIS('PATH1','PATH2,...) opens .txt files at PATH1, PATH2, etc.
%
%    This is a program which takes in .txt files from the activity monitors
%    this lab seems to love so much and creates actograms from them. The
%    format is fairly consistent across the actograms, so if the script
%    doesn't work, make sure there haven't been any major changes.
%
%    If the argument field is left blank, this will bring you to the GUI so
%    you can click the file(s) you want. Otherwise, type the path of the files
%    you want to analyze here.
%
% - SCT 07/18/2014

%% Importing the data

% First check to see if the vargin field is blank

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
else
    for j = 1:nargin
        dataIn(j) = importdata(vargin(j));
    end
end

% Note that, for the actograms readout I was given, the format is as
% follows:

% filename \t date
% numbins
% duration of bin (minutes)
% time of start
% Bin #1
% Bin #2
% etc.

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
numData = str2double(txtdata(5:end,1:3:end));
clear txtdata;

%% Run the analysis

numDays = floor((numBins.*durs)/(60*24));

for i = 1:size(numData,2)
    % A new figure for every fly
    figure;
    suptitle([filenames{i},char(10), dates{1,i}, ' ', dates{2,i},' ', ...
        num2str(years(i)),char(10)]);
    for k = 1:numDays(i)
        % And a new subplot for every day
        subplot(numDays(i),1,k)
        todaysData = numData(1+(1440*(k-1)):1440*k,i);
        % Now we plot the actual actogram
        foreg = bar(1:1440,todaysData,'k');
        hold on;
        % This is for making a red background during non-sleep
        maxNum = max(todaysData);
        % (actually not sleep times...)
        sleepTimes = (todaysData > 0);
        bar(1:1440, sleepTimes*(maxNum+1),... 
            'FaceColor', [238/255,121/255,159/255],'EdgeColor','none','BarWidth',1);
        xlim([0,1440]);
        ylim([0,maxNum+1]);
        % boring formatting stuff
        set(gca,'TickDir','out');
        hold off;
        pch = get(foreg,'child'); % get patch objects from barseries object
        set(pch,'FaceAlpha',.9); % set transparencies
    end
    xlabel('Minutes into day');
end



