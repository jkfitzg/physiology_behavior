from neo.io import AxonIO
import numpy as np
from scipy.io import loadmat
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.stats import circmean, circstd
from plotting_help import *
import sys, os
import scipy.signal
from bisect import bisect
import cPickle
import math
import pandas as pd
import scipy as sp

#---------------------------------------------------------------------------#

class Phys_Flight():  
    def __init__(self, fname):
        if fname.endswith('.abf'):
            self.basename = ''.join(fname.split('.')[:-1])
            self.fname = fname
        else:
            self.basename = fname
            self.fname = self.basename + '.abf'  #check here for fname type 
                  
    def open_abf(self,exclude_indicies=[]):        
        abf = read_abf(self.fname)              #sampled at 10,000 hz
        
        #added features to exclude specific time intervals
        n_indicies = np.size(abf['x_ch']) #assume all channels have the same sample #s 
        inc_indicies = np.setdiff1d(range(n_indicies),exclude_indicies);
                   
        self.xstim = np.array(abf['x_ch'])[inc_indicies]
        self.ystim = np.array(abf['y_ch'])[inc_indicies]

        self.samples = np.arange(self.xstim.size)  #this is adjusted
        self.t = self.samples/float(10000)
 
        #now process wing signal
        lwa_v = np.array(abf['wba_l'])[inc_indicies]
        rwa_v = np.array(abf['wba_r'])[inc_indicies]
        
        lwa_degrees = process_wings(lwa_v)
        rwa_degrees = process_wings(rwa_v)
        
        #self.lwa = np.array(abf['wba_l'])[inc_indicies]
        #self.rwa = np.array(abf['wba_r'])[inc_indicies]
        self.lwa = lwa_degrees
        self.rwa = rwa_degrees
        self.lmr = self.lwa - self.rwa
        self.ao = np.array(abf['patid'])[inc_indicies]
        
        self.vm = np.array(abf['vm'])[inc_indicies] - 13 #offset for bridge potential
        self.tach = np.array(abf['tach'])[inc_indicies]
            
    def _is_flying(self, start_i, stop_i,flying_time_thresh=0.95):  #fix this critera
        #check that animal is flying using the tachometer signal
        
        #nonflight is ~ -.5 V, with an envelope <1, while flight has ~4 V envelope
        #this works but isn't ideal. 
        wba_thres = .4 
        processed_tach_signal = moving_average(abs(self.tach[start_i:stop_i]),1000)
        
        n_flying_samples = np.size(np.where(processed_tach_signal > wba_thres))
        total_samples = stop_i-start_i
        
        is_flying = (float(n_flying_samples)/total_samples) > flying_time_thresh   
        return is_flying
        
           
#---------------------------------------------------------------------------#

class Looming_Phys(Phys_Flight):
    
    def process_fly(self,ex_i=[]):  #does this interfere with the Flight_Phys init?
        self.open_abf(ex_i)
        self.parse_trial_times()
        self.parse_stim_type()
        
    def remove_non_flight_trs(self, iti=750):
        #loop through each trial and determine whether fly was flying continuously
        #delete the non-flight trials
        #directly -- things to change: n_trs, tr_starts, tr_stops, looming stim on
        
        non_flight_trs = [];
        
        for tr_i in range(self.n_trs):
            this_tr_start = self.tr_starts[tr_i] - iti
            this_tr_stop = self.tr_stops[tr_i] + iti
            
            if not self._is_flying(this_tr_start,this_tr_stop):
                non_flight_trs.append(tr_i) 
        
        #print 'nonflight trials : ' + ', '.join(str(x) for x in non_flight_trs)
        
        #now remove these
        self.n_trs = self.n_trs - np.size(non_flight_trs)
        self.tr_starts = np.delete(self.tr_starts,non_flight_trs)  #index values of starting and stopping
        self.tr_stops = np.delete(self.tr_stops,non_flight_trs)
        #self.pre_loom_stim_ons = np.delete(self.pre_loom_stim_ons,non_flight_trs)
                    
    def parse_trial_times(self):
        #parse the ao signal to determine trial start and stop index values
        #include checks for unusual starting aos, early trial ends, 
        #long itis, etc
        
        ao_diff = np.diff(self.ao)
        
        tr_start = self.samples[np.where(ao_diff > 5)]
        start_diff = np.diff(tr_start)
        redundant_starts = tr_start[np.where(start_diff < 1000)]
        clean_tr_starts = np.setdiff1d(tr_start,redundant_starts)+1
        
        tr_stop = self.samples[np.where(ao_diff <= -4)]
        stop_diff = np.diff(tr_stop)
        redundant_stops = tr_stop[np.where(stop_diff < 1000)] 
        #now check that the y value is > 0 
        clean_tr_stop_candidates = np.setdiff1d(tr_stop,redundant_stops)+1
        
        clean_tr_stops = clean_tr_stop_candidates[np.where(self.ao[clean_tr_stop_candidates-5] > 0)]
        
        #check that first start is before first stop
        if clean_tr_stops[0] < clean_tr_starts[0]: 
            clean_tr_stops = np.delete(clean_tr_stops,0)
         
        #last stop is after last start
        if clean_tr_starts[-1] > clean_tr_stops[-1]:
            clean_tr_starts = np.delete(clean_tr_starts,len(clean_tr_starts)-1)
         
        #should check for same # of starts and stops
        n_trs = len(clean_tr_starts)
        
        ##for debugging
        #figd = plt.figure()
        #plt.plot(self.ao)
        #plt.plot(ao_diff,color=magenta)
        #y_start = np.ones(len(clean_tr_starts))
        #y_stop = np.ones(len(clean_tr_stops))
        #plt.plot(clean_tr_starts,y_start*7,'go')
        #plt.plot(clean_tr_stops,y_stop*7,'ro')
        
        #detect when the y stim stepped
        ystim_diff = np.diff(self.ystim)
        y_step = self.samples[np.where(ystim_diff > .03)]

        ##now discriminate first stim on for a trial, not looming steps
        #pre_loom_stim = np.zeros(n_trs)
        #for tr_i in range(n_trs):
        #    earliest_t = clean_tr_starts[tr_i] - 2000  #look within a window before looming
        #    latest_t = clean_tr_starts[tr_i] - 300 
        #    win1 = np.where(y_step > earliest_t)[0]
        #    win2 = np.where(y_step < latest_t)[0]
        #    candidate_stim_starts = y_step[np.intersect1d(win1,win2)]
        #    pre_loom_stim[tr_i] = candidate_stim_starts[0]
        
        #next encode post loom stim on, iti_dur as the median
        
        self.n_trs = n_trs 
        self.tr_starts = clean_tr_starts  #index values of starting and stopping
        self.tr_stops = clean_tr_stops
        #self.pre_loom_stim_ons = pre_loom_stim
        
        #here remove all trials in which the fly is not flying. 
        self.remove_non_flight_trs()
        
    def parse_stim_type(self):
        #calculate the stimulus type
        
        #self.n_trs = 70 #hack to deal with the crash ------ remove this *******************
       
        stim_types_labels = []
        stim_types_labels.append('left, 22 l/v')
        stim_types_labels.append('left, 44 l/v') 
        stim_types_labels.append('left, 88 l/v')
        
        stim_types_labels.append('center, 22 l/v')
        stim_types_labels.append('center, 44 l/v')
        stim_types_labels.append('center, 88 l/v')
        
        stim_types_labels.append('right, 22 l/v')
        stim_types_labels.append('right, 44 l/v')
        stim_types_labels.append('right, 88 l/v')
        
        stim_types = -1*np.ones(self.n_trs,'int')
        
        tr_ao_codes = np.empty(self.n_trs)
        
        #first loop through to get the unique ao values
        for tr in range(self.n_trs): 
            this_start = self.tr_starts[tr]
            this_stop = self.tr_stops[tr]
            tr_ao_codes[tr] = round(np.mean(self.ao[this_start:this_stop]),1)   
        unique_tr_ao_codes = np.unique(tr_ao_codes) 
        
        for tr in range(self.n_trs): 
            tr_ao_code = tr_ao_codes[tr]         
            stim_types[tr] = int(np.where(unique_tr_ao_codes == tr_ao_code)[0][0])
            #this crashes when it gets to the crashed area
                
        
        self.stim_types = stim_types  #change to integer, although nans are also useful
        self.stim_types_labels = stim_types_labels
           
    def plot_wba_stim(self,title_txt=[],wba_lim=[-60,60],if_save=True):
        #plot the stimuli and wba traces for each of the nine conditions
        #add text to annotate the filename, genotype
        #ideally there would be less whitespace between the wba and stim figures and more 
        #between different conditions. check how to do this.
        
        
        fig = plt.figure(figsize=(16.5, 9))
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
        
        cnds_to_plot=range(9)
    
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3))
            grid_col = int(cnd%3)
        
            this_cnd_trs = np.where(self.stim_types == cnd)[0]
            n_cnd_trs = np.size(this_cnd_trs)
            
            #get colormap info
            cmap = plt.cm.get_cmap('jet')     #('gist_ncar')
            cNorm  = colors.Normalize(0,n_cnd_trs)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
            
            s_iti = 5000   #add iti periods
            baseline_win = [500,950]  #be careful not to average out the visual transient here.
               
            x_lim = [0, 3]
               
            for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):
                this_start = self.tr_starts[tr] - s_iti
                this_stop =  self.tr_stops[tr] + s_iti
                this_color = scalarMap.to_rgba(i)        
                
                #plot WBA signal -----------------------------------------------------------    
                wba_ax = plt.subplot(gs[grid_row,grid_col])     
                wba_trace = self.lmr[this_start:this_stop]
                baseline = np.nanmean(wba_trace[baseline_win])
                wba_trace = wba_trace - baseline
                
                wba_ax.plot(self.t[this_start:this_stop]-self.t[this_start],wba_trace,color=this_color)
                
                #plot black line for 0 -------------------------------
                wba_ax.axhline(color=black)
                
                #set x and y lim -------------------------------
                wba_ax.set_ylim(wba_lim) 
                wba_ax.set_xlim(x_lim) 
                
                #remove extra grid labels -------------------------------
                if grid_row == 0 and grid_col == 0:
                    wba_ax.yaxis.set_ticks(wba_lim)
                    wba_ax.set_ylabel('L-R WBA (Degrees)')
                else:
                    wba_ax.yaxis.set_ticks([])
                wba_ax.xaxis.set_ticks([])
                      
                #now plot stimulus traces -----------------------------------------------------
                stim_ax = plt.subplot(gs[grid_row+1,grid_col])
                stim_ax.plot(self.t[this_start:this_stop]-self.t[this_start],self.ystim[this_start:this_stop],color=this_color)
                
                #set x and y lim -------------------------------
                stim_ax.set_xlim(x_lim)
                stim_ax.set_ylim([0, 10]) 
                
                #remove extra grid labels -------------------------------
                if grid_row == 4:
                    stim_ax.xaxis.set_ticks(x_lim)
                    if grid_col == 0:
                        #stim_ax.set_ylabel('Visual stimulus')
                        stim_ax.set_xlabel('Time (s)') 
                else:
                    stim_ax.xaxis.set_ticks([])
                stim_ax.yaxis.set_ticks([])
                stim_ax.set_xlim(x_lim)
        
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw()
        
        if if_save:
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + title_txt + '_looming_wings.png',dpi=100)
            #plt.close('all')
    
    def plot_vm_wba_stim(self,title_txt='',vm_base_subtract = True,l_div_v_list=[0,1,2],
        vm_lim=[-90,-40],wba_lim=[-60,60],if_save=False): 
        #for each l/v stim parameter, 
        #make figure three rows of signals -- vm, wba, stimulus x
        #three columns of looming direction
         
         
        l_div_v_txt = [];
        l_div_v_txt.append('22 l div v')
        l_div_v_txt.append('44 l div v')
        l_div_v_txt.append('88 l div v')
        
        for loom_speed in l_div_v_list:          
            fig = plt.figure(figsize=(16.5, 9))
            gs = gridspec.GridSpec(3,3,width_ratios=[1,1,1],height_ratios=[1,1,.2])
        
            #0 1 2 ; 3 4 5 ; 6 7 8
            cnds_to_plot = np.arange(0,7,3) + loom_speed 
    
            for cnd, grid_col in zip(cnds_to_plot,range(3)):
                #here row is manually set -- corresponds to the signal
                #column is in the loop
                
                this_cnd_trs = np.where(self.stim_types == cnd)[0]
                n_cnd_trs = np.size(this_cnd_trs)
            
                #get colormap info
                cmap = plt.cm.get_cmap('jet')     #('gist_ncar')
                cNorm  = colors.Normalize(0,n_cnd_trs)
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
            
                s_iti = 20000   #add iti periods
                baseline_win = [0,5000]  #be careful not to average out the visual transient here.
               
                #.5, 1, 2s
                x_lim = [0, 4+loom_speed]
               
                for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):
                    this_start = self.tr_starts[tr] - s_iti
                    this_stop =  self.tr_stops[tr] + s_iti
                    this_color = scalarMap.to_rgba(i)        
                    
                    #plot Vm signal ______________________________________________________
                    vm_ax = plt.subplot(gs[0,grid_col])
                    
                    vm_trace = self.vm[this_start:this_stop]
                    if vm_base_subtract:
                        vm_base = np.nanmean(vm_trace[baseline_win])
                        vm_trace = vm_trace - vm_base
                        vm_lim = [-15, 15]
                    
                    vm_ax.plot(self.t[this_start:this_stop]-self.t[this_start],vm_trace,color=this_color)
                
                    #set x and y lim
                    vm_ax.set_ylim(vm_lim) 
                    vm_ax.set_xlim(x_lim) 
                
                    #remove extra grid labels
                    if grid_col == 0:
                        vm_ax.yaxis.set_ticks(vm_lim)
                        if vm_base_subtract:
                            vm_ax.set_ylabel('Baseline subtracted Vm (mV)')
                        else:
                            vm_ax.set_ylabel('Vm (mV)')
                    else:
                        vm_ax.yaxis.set_ticks([])
                    vm_ax.xaxis.set_ticks([])
                    
    
                    #plot WBA signal _____________________________________________________    
                    wba_ax = plt.subplot(gs[1,grid_col])     
                    wba_trace = self.lmr[this_start:this_stop]
                    baseline = np.nanmean(wba_trace[baseline_win])
                    wba_trace = wba_trace - baseline
                
                    wba_ax.plot(self.t[this_start:this_stop]-self.t[this_start],moving_average(wba_trace,200),color=this_color)
                
                    #plot black line for 0 
                    wba_ax.axhline(color=black)
                
                    #set x and y lim
                    wba_ax.set_ylim(wba_lim) 
                    wba_ax.set_xlim(x_lim) 
                
                    #remove extra grid labels
                    if grid_col == 0:
                        wba_ax.yaxis.set_ticks(wba_lim)
                        wba_ax.set_ylabel('L-R WBA (Degrees)')
                    else:
                        wba_ax.yaxis.set_ticks([])
                    wba_ax.xaxis.set_ticks([])
    
                      
                    #now plot stimulus traces ____________________________________________
                    stim_ax = plt.subplot(gs[2,grid_col])
                    stim_ax.plot(self.t[this_start:this_stop]-self.t[this_start],self.ystim[this_start:this_stop],color=this_color)
                
                    #set x and y lim
                    stim_ax.set_xlim(x_lim)
                    stim_ax.set_ylim([0, 10]) 
                
                    #remove extra grid labels
                    if grid_col == 0:
                        stim_ax.xaxis.set_ticks(x_lim)
                        stim_ax.set_ylabel('Visual stimulus')
                        stim_ax.set_xlabel('Time (s)') 
                        stim_ax.yaxis.set_ticks([])
                    else:
                        stim_ax.xaxis.set_ticks([])
                        stim_ax.yaxis.set_ticks([])
                    stim_ax.set_xlim(x_lim)
        
            #now annotate -- signals x positions     
            fig.text(.22,.905,'Left',fontsize=14)
            fig.text(.495,.905,'Center',fontsize=14)
            fig.text(.775,.905,'Right',fontsize=14)
        
            figure_txt = title_txt + ' '+l_div_v_txt[loom_speed]
            fig.text(.425,.95,figure_txt,fontsize=18)        
            plt.draw()
            
            if if_save:
                saveas_path = '/Users/jamie/bin/figures/'
                plt.savefig(saveas_path + figure_txt + '_looming_vm_wings.png',dpi=100)
                #plt.close('all')        
        
    def plot_vm_wba_stim_corr(self,title_txt='',vm_base_subtract = False,l_div_v_list=[1],
        vm_lim=[-80,-50],wba_lim=[-45,45],if_save=False): 
        #for each l/v stim parameter, 
        #make figure four rows of signals -- vm, wba, stimulus, vm-wba corr x
        #three columns of looming direction
        
        #labels for looming conditions 
        l_div_v_txt = [];
        l_div_v_txt.append('22 l div v')
        l_div_v_txt.append('44 l div v')
        l_div_v_txt.append('88 l div v')
        
        #time windows in which to examine turning behaviors. these are by eye
        sampling_rate = 10000
        l_div_v_turn_windows = []
        l_div_v_turn_windows.append(range(int(2.45*sampling_rate),int(2.8*sampling_rate)))
        l_div_v_turn_windows.append(range(int(2.95*sampling_rate),int(3.3*sampling_rate)))
        l_div_v_turn_windows.append(range(int(3.85*sampling_rate),int(4.20*sampling_rate)))
        
        s_iti = 20000   #add iti periods
        baseline_win = range(0,5000)  #be careful not to average out the visual transient here.
        
        #get all traces __________________________________________________________________
        all_fly_traces = self.get_traces_by_stim() 
        
        #now plot one figure for each looming speed ______________________________________
        for loom_speed in l_div_v_list: 
            fig = plt.figure(figsize=(16.5, 9))
            gs = gridspec.GridSpec(4,3,width_ratios=[1,1,1],height_ratios=[1,1,.2,.5])
        
            #store all subplots for formatting later           
            all_vm_ax = []
            all_wba_ax = []
            all_stim_ax = []
            all_corr_ax = []
        
            cnds_to_plot = np.arange(0,7,3) + loom_speed
            #0 1 2 ; 3 4 5 ; 6 7 8 
            this_turn_win = l_div_v_turn_windows[loom_speed]
            
            #now loop through the conditions/columns. ____________________________________
            #the signal types are encoded in separate rows(vm, wba, stim, corr)
            for cnd, grid_col in zip(cnds_to_plot,range(3)):
            
                this_cnd_trs = np.where(self.stim_types == cnd)[0] #rewrite this so I just use all_fly_traces *******************************
                n_cnd_trs = np.size(this_cnd_trs)
            
                #get colormap info _______________________________________________________
                cmap = plt.cm.get_cmap('jet')     #('gist_ncar')
                cNorm  = colors.Normalize(0,n_cnd_trs)
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
            
                #create subplots _________________________________________________________              
                if grid_col == 0:
                    vm_ax = plt.subplot(gs[0,grid_col])
                    wba_ax = plt.subplot(gs[1,grid_col],sharex=vm_ax) 
                    stim_ax = plt.subplot(gs[2,grid_col],sharex=vm_ax)    
                    corr_ax = plt.subplot(gs[3,grid_col],sharex=vm_ax)        
                else:
                    vm_ax = plt.subplot(gs[0,grid_col],sharey=all_vm_ax[0])
                    wba_ax = plt.subplot(gs[1,grid_col], sharex=vm_ax,sharey=all_wba_ax[0]) 
                    stim_ax = plt.subplot(gs[2,grid_col],sharex=vm_ax,sharey=all_stim_ax[0])    
                    corr_ax = plt.subplot(gs[3,grid_col],sharex=vm_ax,sharey=all_corr_ax[0])
                all_vm_ax.append(vm_ax)
                all_wba_ax.append(wba_ax) 
                all_stim_ax.append(stim_ax)
                all_corr_ax.append(corr_ax)
            
                #loop single trials and plot all signals _________________________________
                for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):
                    this_start = self.tr_starts[tr] - s_iti  #change this to just use the pandas df******
                    this_stop  = self.tr_stops[tr] + s_iti
                    trace_t = self.t[this_start:this_stop]-self.t[this_start]
                    this_color = scalarMap.to_rgba(i)        
                    
                    #plot Vm signal ______________________________________________________      
                    vm_trace = self.vm[this_start:this_stop]  #change *********
                    vm_base = np.nanmean(vm_trace[baseline_win])
                    if vm_base_subtract:
                        vm_trace = vm_trace - vm_base
                 
                    vm_ax.plot(trace_t,vm_trace,color=this_color)
                   
                    #plot WBA signal _____________________________________________________           
                    wba_trace = self.lmr[this_start:this_stop]
                    baseline = np.nanmean(wba_trace[baseline_win])
                    wba_trace = wba_trace - baseline  #always subtract the baseline here
                    
                    wba_ax.plot(trace_t,moving_average(wba_trace,200),color=this_color)
                
                    #now plot stimulus traces ____________________________________________
                    stim_ax.plot(trace_t,self.ystim[this_start:this_stop],color=this_color)
                 
                #calculate, plot correlations for all traces/cnd _________________________             
                vm_baseline = np.nanmean(all_fly_traces.loc[baseline_win,('this_fly',slice(None),cnd,'vm')],0)
                lmr_turn    = abs(np.nanmean(all_fly_traces.loc[this_turn_win,('this_fly',slice(None),cnd,'lmr')],0))
                 
                t_steps = range(0,39000,1000)  #update this for all conditions *********
                step_size = 10000
                
                for t_start in t_steps:
                    t_stop = t_start+step_size
                    this_vm = np.nanmean(all_fly_traces.loc[:,('this_fly',slice(None),cnd,'vm')][t_start:t_stop],0)
                    non_nan = np.where(~np.isnan(lmr_turn))[0]  #is there a more elegant way to do this? 
                    delta_vm = this_vm-vm_base
                    
                    r,p = sp.stats.pearsonr(delta_vm[non_nan],lmr_turn[non_nan])
                    t_plot = (t_start+(step_size/2.0))/sampling_rate
                    corr_ax.plot(t_plot,r,'.b')
                    if p < 0.05: #if significant without correcting for many comparisons
                        l = corr_ax.plot(t_plot,r,'or',) #plot in red
                
                        
            #now format all subplots _____________________________________________________  
            
            vm_lim = vm_ax.get_ylim()
            wba_lim = wba_ax.get_ylim()
            
            #loop though all columns again, format each row ______________________________
            for col in range(3):    
            
                #create shaded regions of baseline vm and saccade time ___________________
                all_vm_ax[col].fill([0,.5,.5,0],[vm_lim[1],vm_lim[1],vm_lim[0],vm_lim[0]],'black',alpha=.1)
                
                wba_min_t = l_div_v_turn_windows[loom_speed][0]/np.double(sampling_rate)
                wba_max_t = l_div_v_turn_windows[loom_speed][-1]/np.double(sampling_rate)
                all_wba_ax[col].fill([wba_min_t,wba_max_t,wba_max_t,wba_min_t],
                        [wba_lim[1],wba_lim[1],wba_lim[0],wba_lim[0]],'black',alpha=.1)
                        
                #set the ylim for the stimulus and correlation rows ______________________
                all_stim_ax[col].set_ylim([0,10])
                all_corr_ax[col].set_ylim([-1,1])
                 
                #label axes, show xlim and ylim __________________________________________
                
                #remove all time xticklabels
                all_vm_ax[col].tick_params(labelbottom='off')
                all_wba_ax[col].tick_params(labelbottom='off')
                all_stim_ax[col].tick_params(labelbottom='off')
                all_corr_ax[col].tick_params(labelbottom='off')
                
                
                if col == 0: #label yaxes
                    all_vm_ax[col].set_ylabel('Vm (mV)')
                    all_wba_ax[col].set_ylabel('WBA (V)')
                    all_stim_ax[col].set_ylabel('Stim (frame)')
                    all_corr_ax[col].set_ylabel('Corr(Vm, WBA)')
                    
                    #label time x axis for just col 0
                    all_corr_ax[col].tick_params(labelbottom='on')
                    all_corr_ax[col].set_xlabel('Time (s)') 

                else: #remove all ylabels 
                    all_vm_ax[col].tick_params(labelleft='off')
                    all_wba_ax[col].tick_params(labelleft='off')
                    all_stim_ax[col].tick_params(labelleft='off')
                    all_corr_ax[col].tick_params(labelleft='off')
                 
            #now annotate stimulus positions, title ______________________________________      
            fig.text(.22,.905,'Left',fontsize=14)
            fig.text(.495,.905,'Center',fontsize=14)
            fig.text(.775,.905,'Right',fontsize=14)
            
            figure_txt = title_txt + ' '+l_div_v_txt[loom_speed]
            fig.text(.425,.95,figure_txt,fontsize=18)        
                   
            plt.draw()
            
            if if_save:
                saveas_path = '/Users/jamie/bin/figures/'
                plt.savefig(saveas_path + figure_txt + '_looming_vm_wings_corr.png',dpi=100)    
   
   
   
    def get_traces_by_stim(self,fly_name='this_fly'):
    #here extract the traces for each of the stimulus times. 
    #align to looming start, and add the first pre stim and post stim intervals
    #here return a data frame of lwa and rwa wing traces
    #self.stim_types already holds an np.array vector of the trial type indicies
   
    #using a pandas data frame with multilevel indexing! rows = time in ms
    #columns are multileveled -- genotype, fly, trial index, trial type, trace
        
        pre_loom_stim_dur = 10000 #add this to the flies? 
        fly_df = pd.DataFrame()
        
        for tr in range(self.n_trs):
            this_loom_start = self.tr_starts[tr]
            this_start = this_loom_start - 25000
            this_stop = self.tr_stops[tr] + pre_loom_stim_dur
            
            this_stim_type = self.stim_types[tr]
            iterables = [[fly_name],
                         [tr],
                         [this_stim_type],
                         ['lmr','lwa','rwa','vm','ystim']]
            column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type','trace'])
            
            
            
            tr_traces = np.asarray([self.lmr[this_start:this_stop],
                                         self.lwa[this_start:this_stop],
                                         self.rwa[this_start:this_stop],
                                         self.vm[this_start:this_stop],
                                         self.ystim[this_start:this_stop]]).transpose()  #reshape to avoid transposing
                                          
            tr_df = pd.DataFrame(tr_traces,columns=column_labels) #,index=time_points)
            fly_df = pd.concat([fly_df,tr_df],axis=1)
         
        return fly_df  
        
    

#---------------------------------------------------------------------------#
def moving_average(values, window):
    #next add gaussian, kernals, etc
    #pads on either end to return an equal length structure,
    #although the edges are distorted
    
    if (window % 2): #is odd 
        window = window + 1; 
    halfwin = window/2
    
    n_values = np.size(values)
    
    padded_values = np.ones(n_values+window)*np.nan
    padded_values[0:halfwin] = np.ones(halfwin)*np.mean(values[0:halfwin])
    padded_values[halfwin:halfwin+n_values] = values
    padded_values[halfwin+n_values:window+n_values+1] = np.ones(halfwin)*np.mean(values[-halfwin:n_values])
  
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(padded_values, weights, 'valid')
    return sma[0:n_values]
        
def xcorr(a, v):
    a = (a - np.mean(a)) / (np.std(a) * (len(a)-1))
    v = (v - np.mean(v)) /  np.std(v)
    xc = np.correlate(a, v, mode='same')
    return xc
    
def read_abf(abf_filename):
        fh = AxonIO(filename=abf_filename)
        segments = fh.read_block().segments
    
        if len(segments) > 1:
            print 'More than one segment in file.'
            return 0

        analog_signals_ls = segments[0].analogsignals
        analog_signals_dict = {}
        for analog_signal in analog_signals_ls:
            analog_signals_dict[analog_signal.name.lower()] = analog_signal

        return analog_signals_dict
        
def process_wings(raw_wings):
    #here shift wing signal -12 ms in time, filling end with nans
    shifted_wings = np.empty_like(raw_wings)
    shifted_wings[:] = np.nan
    shifted_wings[0:-12] = raw_wings[12:]   
    
    #now multiply to convert volts to degrees
    processed_wings = -45 + shifted_wings*33.75
        
    #also look for dropped wing signals _________________________________
    
    artifacts = np.where(abs(np.diff(processed_wings)) > 30)[0]-1
    
    artifact_diff = np.diff(artifacts)
    small_diff_i = np.where(artifact_diff < .2*10000)[0]
    
    #now connect the dots between nearby artifacts
    #for each point in to_connect_times, find the next point, connect these
    
    to_connect_is = small_diff_i
    filled_is = []

    for i in range(np.size(to_connect_is)-1):
        connect_start_artifact_i = to_connect_is[i]
        i_start = artifacts[connect_start_artifact_i]
        i_stop = artifacts[connect_start_artifact_i + 1] #find the artifact closest after this.
        filled_is = np.concatenate([filled_is,\
                                    np.arange(i_start,i_stop)])
    
    if np.size(filled_is) > 0:
        #now loop through a few offsets
        for offset in range(-2,3):      
            a = 5                   
            #processed_wings[filled_is.astype(int)+offset] = np.nan          
            
    #now filter wings   
    #processed_wings = filter_wings(processed_wings)                  
     
    #for debugging
    #calculate or pass in t
    #plt.plot(t[0:-1],np.diff(raw_wings),'green')
    #plt.plot(t[artifacts],40*np.ones_like(artifacts),'*b')
    #plt.plot(t[artifacts[small_diff_i]],35*np.ones_like(small_diff_i),'.m')
    #plt.plot(t[filled_is.astype(int)],37.5*np.ones_like(filled_is),'*b')
            
    return processed_wings
    
def filter_wings(raw_trace):
    # Filter requirements.
    order = 8
    fs = 1000         # sample rate, Hz
    cutoff = 6  # desired cutoff frequency of the filter, Hz
    y = butter_lowpass_filter(raw_trace, cutoff, fs, order)
    y_shifted = np.copy(y)
    y_shifted[:-160] = y[160:]
    y_shifted[0:160] = np.nan
    filtered_trace = y_shifted
    
    return filtered_trace
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sp.signal.lfilter(b, a, data)
    return y
      
def write_to_pdf(f_name,figures_list):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(fname)
    for f in figures_list:
        pp.savefig(f)
    pp.close()

def plot_many_flies(path_name, filenames_df):    

    #loop through all genotypes
    genotypes = (pd.unique(filenames_df.values[:,1]))
    print genotypes
    
    for g in genotypes:
        these_genotype_indicies = np.where(filenames_df.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + filenames_df.values[index,0])
            title_txt = filenames_df.values[index,1] + '  ' + filenames_df.values[index,0]
            fly.process_fly()
            fly.plot_wba_stim(title_txt)
        
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + title_txt + '_kir_looming.png',dpi=100)
            plt.close('all')
                    
def get_pop_traces_df(path_name, population_f_names):  
    #loop through all genotypes
    #structure row = time points, aligned to looming start
    #columns: genotype, fly, trial index, trial typa, lwa/rwa
    #just collect these for all flies
    
    #genotypes must be sorted to the labels for columns 
    genotypes = (pd.unique(population_f_names.values[:,1]))
    genotypes = np.sort(genotypes)
    genotypes = genotypes[1:]
    print genotypes
    
    population_df = pd.DataFrame()
    
    #loop through each genotype  
    for g in genotypes:
        g
        these_genotype_indicies = np.where(population_f_names.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + population_f_names.values[index,0])
            fly.process_fly()
            fly_df = fly.get_traces_by_stim(g)
            population_df = pd.concat([population_df,fly_df],axis=1)
    return population_df
     
def plot_pop_flight_behavior_histograms(population_df, wba_lim=[-3,3],cnds_to_plot=range(9)):  
    #for the looming data, plot histograms over time of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_cell_names = population_df.loc[:,(g)].columns.get_level_values(0)
        n_cells = np.size(np.unique(all_cell_names))
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
     
            #plot WBA histogram signal -----------------------------------------------------------    
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
        
            #just plot the mean for debugging
            #wba_ax.plot(np.nanmean(g_lmr,1))
        
            #now plot the histograms over time. ------------
            max_t = np.shape(g_lmr)[0]
            n_trs = np.shape(g_lmr)[1]
                     
            t_points = range(max_t)
            t_matrix = np.tile(t_points,(n_trs,1))
            t_matrix_t = np.transpose(t_matrix)

            t_flat = t_matrix_t.flatten() 
            g_lmr_flat = g_lmr.flatten()

            #now remove nans
            g_lmr_flat = g_lmr_flat[~np.isnan(g_lmr_flat)]
            t_flat = t_flat[~np.isnan(g_lmr_flat)]

            #calc, plot histogram
            h2d, xedges, yedges = np.histogram2d(t_flat,g_lmr_flat,bins=[200,50],range=[[0, 4200],[-3,3]],normed=True)
            wba_ax.pcolormesh(xedges, yedges, np.transpose(h2d))
        
           
            #plot white line for 0 -----------
            wba_ax.axhline(color=white)
        
            wba_ax.set_xlim(x_lim) 
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_histograms.png',dpi=100)
        #plt.close('all')

def plot_pop_flight_behavior_means(population_df, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=black,linewidth=.5)        
        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=magenta,linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)
        
            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_means.png',dpi=100)
        plt.close('all')
        
def plot_pop_flight_behavior_means_overlay(population_df, two_genotypes, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    fig = plt.figure(figsize=(16.5, 9))
    #change this so I'm not hardcoding the number of axes
    gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
    genotype_colors = [magenta, blue]
    
    i = 0 
    title_txt = '';
    for g in two_genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = title_txt + g + ' __ ' + str(n_cells) + ' flies ' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
        
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=.25)        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)

            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)

            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
          
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])

            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=black)

            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 

            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        i = i + 1
        
    #now annotate        
    fig.text(.06,.8,'left',fontsize=14)
    fig.text(.06,.53,'center',fontsize=14)
    fig.text(.06,.25,'right',fontsize=14)
    
    fig.text(.22,.905,'22 l/v',fontsize=14)
    fig.text(.495,.905,'44 l/v',fontsize=14)
    fig.text(.775,.905,'88 l/v',fontsize=14)        

    fig.text(.1,.95,two_genotypes[0],color='magenta',fontsize=18)
    fig.text(.2,.95,two_genotypes[1],color='blue',fontsize=18)
    plt.draw()
    
    saveas_path = '/Users/jamie/bin/figures/'
    plt.savefig(saveas_path + title_txt + '_population_kir_looming_means_overlay_' 
        + two_genotypes[0] + '_' + two_genotypes[1] + '.png',dpi=100)
    #plt.close('all')


    
