#!/bin/bash

#Load Matlab 2017a module
ml MATLAB
matlab -singleCompThread -nojvm -nodisplay<< EOF

% define scale/parcellation
disp('computing model')

% Schaefer2018_200Parcels_17Networks_order
% Glasser_MMP
% aal2002_90
parcellation='Schaefer2018_300Parcels_17Networks_order'
dataset='hcp'

% define bin
binmod = 'multibin'

%cd('/home/kobelevax/Studien/spatial_scales/model')
path=pwd;
addpath(path);
path2=[pwd '/Brain_songs'];
addpath(path2);
%rng('default');
%r=rand(1000,100);

matrix_index = str2num(getenv('SLURM_ARRAY_TASK_ID'))
subs=1:1;
sub=subs(matrix_index)
disp('current subject')


no_of_subs=sub;
tempentro=0;


% define global coupling parameter (based on fitting)
if dataset=='nki'
	if parcellation=='Schaefer2018_100Parcels_17Networks_order' 
		we= 1.55
	elseif parcellation=='Schaefer2018_200Parcels_17Networks_order'
	we= 1.525
	elseif parcellation=='Schaefer2018_300Parcels_17Networks_order'
	we= 1.85
	elseif parcellation=='Schaefer2018_400Parcels_17Networks_order'
	we= 1.8
	end
	end

	if dataset=='hcp'
		if parcellation=='Schaefer2018_100Parcels_17Networks_order'
			we= 1.6
		elseif parcellation=='Schaefer2018_200Parcels_17Networks_order'
		we= 1.575
		elseif parcellation=='Schaefer2018_300Parcels_17Networks_order'
		we= 1.85
		elseif parcellation=='Schaefer2018_400Parcels_17Networks_order'
		we= 1.875
		end
		end

		disp('length of simulated time course')

		if dataset == 'hcp'
			TRms = 720;
		ts = 1200;
		window_size = 80   %window size: 80s
		timemax = 1;
		elseif dataset =='nki'
		TRms = 2000;
		ts = 178;
		timemax = 10;
		window_size = 30   %window size: 30s
		end

		% define length of simulation and size of temporal bins
		TR=TRms/1000
		BIN=10:10:3000;
		Tsec = 2000
		Tmin = Tsec/60
		Tmax= round(Tsec/TR)

		% define, load and normalize structural matrix
		foldername = [pwd '/data/leaddbs_sc/' parcellation '/']; 
		filename = [ 'roi_1_struc_matrix.mat'];
		file       = fullfile(foldername, filename);
		load(file);
		C=mat; 						%rename structural matrix
		N = size(C,1);            %make zeros in diagonal of C
		Isubdiag = find(tril(ones(N),-1));
		for i=1:N
		C(i,i) = 0;
		end
		C=C/max(max(C))*0.2; %normalize structural matrix

		% where to save the resulting file and how to name it
		disp('location of saved file');
		folderj = [pwd '/simulated_DFC/leaddbs/' dataset '/' parcellation]
		mkdir(folderj);
		filesave1= ['brainsongs' num2str(no_of_subs) '_we' num2str(we) '_tsec' num2str(Tsec) '_win' num2str(window_size)  '.mat']
		filesave2= ['neuroact' num2str(no_of_subs) '_we' num2str(we) '_tsec' num2str(Tsec) '_win' num2str(window_size)  '.mat']
		foldersave= [folderj '/sim_entropy']
		savefile = fullfile(foldersave, filesave1)
		savefile2 = fullfile(foldersave, filesave2)
		if (~exist(foldersave));
			mkdir(foldersave);
		end
		cd(foldersave);

		% run simulation (DMF model)
		% definition of parameters
		dtt   = 1e-3;   % Sampling rate of simulated neuronal activity (seconds)
		dt=0.1;

		taon=100;
		taog=10;
		gamma=0.641;
		sigma=0.01;
		JN=0.15;
		I0=0.382;
		Jexte=1.;
		Jexti=0.7;
		w=1.4;

		%%% definition of TR, filters

		flp = .008;              % lowpass frequency of filter
		fhi = .08;               % highpass
		delt = TR;            % sampling interval of empirical FC = TR
		k=2;                    % 2nd order butterworth filter
		fnq=1/(2*delt);         % Nyquist frequency
		Wn=[flp/fnq fhi/fnq];   % butterworth bandpass non-dimensional frequency
		[bfilt2,afilt2]=butter(k,Wn);   % construct the filter

		%%%%%%%%%%%%
		%% Optimize
		%%


		%%run many trials
		disp('balance j')
		J=Balance_J(we,C);
		rng('shuffle');
		neuro_act=zeros(round(1000*(Tmax-1)*2+1),N);
		sn=0.001*ones(N,1);
		sg=0.001*ones(N,1);
		nn=1;
		for t=0:dt:(1000*(Tmax-1)*2)
		xn=I0*Jexte+w*JN*sn+we*JN*C*sn-J.*sg;
		xg=I0*Jexti+JN*sn-sg;
		rn=phie(xn);
		rg=phii(xg);
		sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+sqrt(dt)*sigma*randn(N,1);
		sn(sn>1) = 1;
		sn(sn<0) = 0;
		sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma*randn(N,1);
		sg(sg>1) = 1;
		sg(sg<0) = 0;
		j=j+1;
		if abs(mod(t,1))<0.01
			neuro_act(nn,:)=rn';
		nn=nn+1;
		end
		end
		nn=nn-1;

		disp('friston balloon')

		%%% create BOLD signal from neural model
		% Friston BALLOON MODEL
		T = nn*dtt; % Total time in seconds
		B = BOLD(T,neuro_act(1:nn,1)'); % B=BOLD activity, bf=Foutrier transform, f=frequency range)
		BOLD_act = zeros(length(B),N);
		BOLD_act(:,1) = B;
		for nnew=2:N
		B = BOLD(T,neuro_act(1:nn,nnew));
		BOLD_act(:,nnew) = B;
		end
		bds=BOLD_act((TR*1000):(TR*1000):end,:);
		neuro_act=neuro_act(3000:end,:);
		disp('model calculated');

		%%run many trials
		ii=1;

		% bin timeseries
		for bin=BIN
		clear events events1 events2 neuro_actbin prob probfinal BSong;
		sum_binsize(ii)=bin;
		neuro_actbin=zeros(ceil(size(neuro_act,1)/10),N);
		j=1;

		for i=1:bin:size(neuro_act,1)-bin;
		neuro_actbin(j,:)=mean(neuro_act(i:i+bin,:),1); %these are non-overlapping windows; old script: overlapping windows
		j=j+1;
		end

		neuro_actbin=neuro_actbin(1:j-1,:);    %non-overlapping windows
		bb=floor(bin/4);
		FC=corrcoef(neuro_actbin);
		FC_bds=corrcoef(bds);
		FCm_bds(ii)=mean(FC_bds(Isubdiag));
		FCm1(ii)=mean(FC(Isubdiag));

		% calculation of other metrics using brain songs
		for seed=1:N
		tise = neuro_actbin(:,seed);
		tise=demean(detrend(tise));
		ev1=tise>std(tise);
		ev2=[0 ev1(1:end-1)'];
		events(seed,:)=(ev1'-ev2)>0;
		end

		% calculate patterns and activity matrix
		Patterns = assembly_patterns(events) ;
		Activities = assembly_activity(Patterns,events);
		sizeact= size(Activities)

		NumPatterns(ii)=size(Patterns,2);
		disp('ntime')
		ntime=size(Activities,2)
		save(savefile2, 'Activities', 'Patterns');   

		% spatial entropy of patterns
		for j=1:NumPatterns(ii)
		disp('prob')
		prob(j)=sum(Activities(j,:))/double(ntime)
		end
		for j=1:NumPatterns(ii)
		disp('probfinal')
		probfinal(j)=prob(j)/sum(prob)
		probdiff_old(j)= 1/NumPatterns(ii).*(probfinal(j)-1/NumPatterns(ii))
		probdiff(j)= (probfinal(j)-1/NumPatterns(ii)).*(0.5.*NumPatterns(ii))/(NumPatterns(ii)-1)
		end

		Ind=find(probfinal~=0);
		entro1(ii)=-sum(probfinal(Ind).*log(probfinal(Ind)));
		disp('entrodiff')
		entrodiff(ii)=1-sum(probdiff)
		entrodiff_old(ii)=1-sum(probdiff_old)

		for i=1:N
		suma=0;
		for j=1:NumPatterns(ii)
		suma=suma+abs(Patterns(i,j))*sum(abs(Patterns(:,j)))*probfinal(j);
		end
		BSong(i)=suma;
		end

		entronorm(ii)= entro1(ii)./NumPatterns(ii); %normalized entropy
		entronormlog(ii)= entro1(ii)./log(NumPatterns(ii)); %normalized entropy


		save(savefile, 'entrodiff', 'entrodiff_old', 'entro1', 'entronorm', 'FCm1', 'FCm_bds', 'sum_binsize', 'NumPatterns', 'entronormlog');   

		ii=ii+1;

		end
		EOF






