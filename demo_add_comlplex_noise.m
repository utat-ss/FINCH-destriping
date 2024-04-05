%% Generate Training Dataset (Stage 3: complex noise)
clc;clear;
rng(0);
addpath(genpath(pwd));

basedir = '/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/data/datasets/ICVL_test_pair_whole_image/stage1/';
datadir = '/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_test/';
namedir = '/Users/prithviseran/Documents/GitHub/GRN/GRN-HSI-Denoising/';
g = load(fullfile(namedir, 'test_fns.mat'));
fns = g.fns;
preprocess = @(x)(rot90(x,1));

% %%% for complex noise (randomly select from case 1 to case 4)
% newdir = fullfile(basedir, ['icvl_','complex','_case1']);
sigmas = [10 30 50 70];
% generate_dataset_mixture_test_case1(datadir, fns, newdir, sigmas, 'rad', preprocess);
% 
% newdir = fullfile(basedir, ['icvl_','complex','_case2']);
% generate_dataset_mixture_test_case2(datadir, fns, newdir, sigmas, 'rad', preprocess);
% 
% newdir = fullfile(basedir, ['icvl_','complex','_case3']);
% generate_dataset_mixture_test_case3(datadir, fns, newdir, sigmas, 'rad', preprocess);

newdir = fullfile(basedir, ['icvl_','complex','_case4']);
generate_dataset_mixture_test_case4(datadir, fns, newdir, sigmas, 'rad', preprocess);

newdir = fullfile(basedir, ['icvl_','complex','_case5']);
generate_dataset_mixture_test_case5(datadir, fns, newdir, sigmas, 'rad', preprocess);
