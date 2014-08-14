function HMM = flyHMM(states, data)
% WARNING: INCOMPLETE
%
% Takes in a cell vector of state structs and finds the most likely parameter
% distribution given the data
%
%   HMM = flyHMM(STATES, DATA) takes in a vector of structs, each of which
%   corresponds to one state in the HMM with each parameter's prior
%   inserted in the struct. It then uses the data DATA to compute the
%   mostly likely parameter values with a Bayesian updating procedure,
%   stopping when an update changes the parameters very minimally (thus
%   assuming it's reached a local maximum in the probability distribution
%   over the parameters).
%   
%   I want to make this function because MATLAB's native HMM trainer is a
%   little bit shitty and has several frustrating limitations. This hopes
%   to avoid those.
%
% - SCT 08/01/2014
    
HMM = states;
numStates = length(states);

% p( params | data ) = p( data | params ) * p( params ) / p( data )
% We ignore the distribution the data absent a model ( p( data ) ) since we
% look only for the arguments of the extrema of p( params | data ), not
% their actual values. 

% For priors, we let 
