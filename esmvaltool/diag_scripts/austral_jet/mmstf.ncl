; #############################################################################
; austral_jet/mmstf.ncl
; #############################################################################
; Author: Manuel Schlund (DLR, Germany)
; #############################################################################
;
; Description:
;     Simple script to calculate and save the zonal mean meridional stream
;     function mmstf for the use in other diagnostics (e.g. GBRT models).
;
; Required diag_script_info attributes:
;     calc_type : Calculation type (mean/trend)
;     tag : Tag for the diagnostic.
;
; Optional diag_script_info attributes:
;     None
;
; Caveats:
;     None
;
; Modification history:
;     20181204_A-schl_ma: written.
;
; #############################################################################

load "interface_scripts/interface.ncl"
load "interface_scripts/interface.ncl"


begin

  enter_msg(DIAG_SCRIPT, "")

  ; Input data
  va_info = select_metadata_by_name(input_file_info, "va")
  ps_info = select_metadata_by_name(input_file_info, "ps")

  ; Check if datasets are equal
  va_datasets = metadata_att_as_array(va_info, "dataset")
  ps_datasets = metadata_att_as_array(ps_info, "dataset")
  if (.not. all(va_datasets .eq. ps_datasets)) then
    error_msg("f", DIAG_SCRIPT, "", "va and ps variables do not contain " + \
              "the same datasets")
  end if
  n_datasets = dimsizes(va_datasets)

  ; Required attributes
  req_atts = (/"calc_type", "tag"/)
  exit_if_missing_atts(diag_script_info, req_atts)

end

begin

  ; Calculate mmstf for every dataset
  do i_dat = 0, n_datasets - 1
    va = read_data(va_info[i_dat])
    ps = read_data(ps_info[i_dat])
    if (dimsizes(va&lat) .ne. dimsizes(ps&lat)) then
      ps := ps({time|:}, {lat|va&lat}, {lon|:})
    end if

    ; Preprocess plev (see documentation of NCL's zonal_mpsi_Wrap)
    pr = va&plev
    p_idx = ind(pr .gt. 500.0 .and. pr .le. 100500.0)
    pr := pr(p_idx)
    vt2b = va(:, p_idx, :, :)
    if (pr(0) .eq. max(pr)) then
      pr := pr(::-1)
      vt2b := vt2b({time|:}, {plev|::-1}, {lat|:}, {lon|:})
    end if

    ; Calculate zonal mean meridional stream function
    outvar = zonal_mpsi_Wrap(vt2b, va&lat, pr, ps)
    outvar!0 = va!0
    outvar!1 = va!1
    outvar!2 = va!2
    outvar&time = va&time
    outvar&plev = va&plev(p_idx)
    outvar&lat = va&lat
    delete([/va, ps, pr, p_idx, vt2b/])

    ; Set attributes
    ; copy_VarAtts(variable_info[i_dat], outvar)
    outvar@var = "mmstf"
    outvar@diag_script = DIAG_SCRIPT
    outvar@standard_name = "zonal_meridional_stream_function"
    outvar@long_name = "Zonal Meridional Stream Function"
    outvar@units = "kg s-1"
    outvar@ncdf = config_user_info@work_dir + "mmstf_" + \
      va_info[i_dat]@dataset + "_" + va_info[i_dat]@ensemble + ".nc"

    ; Save file
    netcdf_outfile = ncdf_write(outvar, outvar@ncdf)
    delete(outvar)
  end do

end

  ; ---------------------------------------------------------------------------
  ; Read recipe and config data
  ; ---------------------------------------------------------------------------

  ; Plot file type
  ; file_type = config_user_info@output_file_type
  ; if (ismissing(file_type)) then
  ;   file_type = "ps"
  ; end if

  ; Season
  ; if(isatt(diag_script_info, "e13fig10_season")) then
  ;   season = diag_script_info@e13fig10_season
  ; else
  ;   if (isStrSubset(field_type0, "z") .eq. False) then
  ;     error_msg("f", diag_script, "", "plot requires attribute " + \
  ;               "'e13fig10_season' in recipe")
  ;   end if
  ; end if

  ; Additional observations
  ; if (isatt(diag_script_info, "rsondes")) then
  ;   if(isatt(diag_script_info, "e13fig10_yr_min")) then
  ;     year_min = diag_script_info@e13fig10_yr_min
  ;   else
  ;     error_msg("f", diag_script, "", "plot requires attribute " + \
  ;               " 'e13fig10_yr_min' in recipe")
  ;   end if
  ;   if(isatt(diag_script_info, "e13fig10_yr_max")) then
  ;     year_max = diag_script_info@e13fig10_yr_max
  ;   else
  ;     error_msg("f", diag_script, "", "plot requires attribute " + \
  ;               " 'e13fig10_yr_max' in recipe")
  ;   end if
  ; end if

  ; Attributes for MDER analysis
  ; if (.not. isatt(diag_script_info, "wdiag")) then
  ;   diag_script_info@wdiag = (/"", ""/)
  ; end if
  ; if (.not. isatt(diag_script_info, "wdiag_title")) then
  ;   diag_script_info@wdiag_title = (/"", ""/)
  ; end if

  ; Get datasets (consider ensemble averaging if desired)
  ; avgens = False
  ; if (isatt(diag_script_info, "average_ens")) then
  ;   avgens = diag_script_info@average_ens
  ; end if
  ; if (avgens) then
  ;   idx_datasets = UNIQ(input_file_info@dataset)
  ; else
  ;   idx_datasets = ispan(0, dimsizes(input_file_info@dataset) - 1, 1)
  ; end if

  ; Array sizes
  ; DIM_DAT = dimsizes(idx_datasets)

  ; Output arrays
  ; val_trend = new((/2, DIM_DAT/), "float")
  ; val_trend!0 = "case"
  ; val_trend&case = (/"val", "stddev"/)
  ; val_trend!1 = "dataset"
  ; val_trend&dataset = input_file_info@dataset(idx_datasets)
  ; val_mean = new((/2, DIM_DAT/), "float")
  ; val_mean!0 = "case"
  ; val_mean&case = (/"val", "stddev"/)
  ; val_mean!1 = "dataset"
  ; val_mean&dataset = input_file_info@dataset(idx_datasets)

  ; ---------------------------------------------------------------------------
  ; Derive variables
  ; ---------------------------------------------------------------------------

  ; if (isatt(diag_script_info, "derive_var")) then
  ;   var1 = diag_script_info@derive_var
  ;   log_info("Deriving variable " + var1)

  ;   Check attributes
  ;   if (isatt(diag_script_info, "derive_latrange")) then
  ;     lat_range = diag_script_info@derive_latrange
  ;   else
  ;     error_msg("f", diag_script, "", "when deriving a variable the " + \
  ;               "attribute derive_latrange is required")
  ;   end if
  ;   if (isatt(diag_script_info, "derive_lev")) then
  ;     lev = diag_script_info@derive_lev
  ;   else
  ;     error_msg("f", diag_script, "", "when deriving a variable the " + \
  ;               "attribute derive_lev is required")
  ;   end if
  ;   log_info("Retrieved pressure level " + lev + " and latitude range " + \
  ;            lat_range(0) + " to " + lat_range(1) + " from recipe")

  ;   Iterate over all datasets
  ;   do i = 0, dimsizes(input_file_info@dataset) - 1

  ;     Read file and skip if already derived
  ;     filename = input_file_info@filename(i)
  ;     f = addfile(filename, "r")
  ;     if (isfilevar(f, var1)) then
  ;       log_info("Variable was already derived, skipping")
  ;       delete(f)
  ;       break
  ;     end if

  ;     Tropopause pressure tpp
  ;     if (var1 .eq. "tpp") then

  ;       Read ta
  ;       A0 = read_data(i, "ta", "T3M")

  ;       Order plev: lowest pressure first (in hPa)
  ;       pr = A0&plev
  ;       pr = pr / 100.0
  ;       if (pr(0) .eq. max(pr)) then
  ;         pin = pr(::-1)
  ;         tin = A0({time|:}, {lat|:}, {lon|:}, {plev|::-1})
  ;       else
  ;         pin = pr
  ;         tin = A0({time|:}, {lat|:}, {lon|:}, {plev|:})
  ;       end if

  ;       Calculate tropopause pressure and copy metedata
  ;       outvar = trop_wmo(pin, tin, 0, False)
  ;       outvar!0 = A0!0
  ;       outvar!1 = A0!2
  ;       outvar!2 = A0!3
  ;       outvar&time = A0&time
  ;       outvar&lat = A0&lat
  ;       outvar&lon = A0&lon
  ;       outvar@var = var1
  ;       outvar@standard_name = "tropopause_air_pressure"
  ;       outvar@long_name = "Tropopause Air Pressure"
  ;       outvar@units = "hPa"
  ;       outvar := area_operations(outvar, lat_range(0), lat_range(1), 0.0, \
  ;                                 360.0, "average", True)
  ;       delete([/A0, pr, pin, tin/])

  ;     Zonal meridional stream function mmstf
      ; else if (var1 .eq. "mmstf") then

      ;   ; Read va and ps
      ;   va = read_data(i, "va", "T3M")
      ;   ps = read_data(i, "ps", "T2Ms")
      ;   if (dimsizes(va&lat) .ne. dimsizes(ps&lat)) then
      ;     ps := ps({time|:}, {lat|va&lat}, {lon|:})
      ;   end if

      ;   ; Preprocess plev (see docu of zonal_mpsi_Wrap)
      ;   pidx = ind(pr.gt.500.0 .and. pr.lt.100500.0)
      ;   pr = va&plev
      ;   pr := pr(pidx)
      ;   vt2b = va(:, pidx, :, :)
      ;   if (pr(0) .eq. max(pr)) then
      ;     pr := pr(::-1)
      ;     vt2b := vt2b({time|:}, {lat|:}, {lon|:}, {plev|::-1})
      ;   end if

      ;   ; Calculate zonal meridional stream function
      ;   outvar = zonal_mpsi_Wrap(vt2b, va&lat, pr, ps)
      ;   outvar!0 = va!0
      ;   outvar!1 = va!1
      ;   outvar!2 = va!2
      ;   outvar&time = va&time
      ;   outvar&plev = va&plev
      ;   outvar&lat = va&lat
      ;   outvar@var = var1
      ;   outvar@standard_name = "zonal_meridional_stream_function"
      ;   outvar@long_name = "Zonal Meridional Stream Function"
      ;   outvar@units = "kg s-1"
      ;   delete([/va, ps, pr, pidx, vt2b/])
      ; end if
      ; end if

  ;     Write variable info
  ;     variable_info@short_name = var1
  ;     variable_info@standard_name = outvar@standard_name
  ;     variable_info@long_name = outvar@long_name
  ;     variable_info@units = outvar@units

  ;     Write file
  ;     outvar@diag_script = diag_script
  ;     ncdf_outfile = ncdf_write(outvar, filename)
  ;     delete([/f, outvar/])
  ;   end do
  ;   var0 := var1

  ; Get plev and lat range from preprocessed data if variable is not derived
  ; else
  ;   f = addfile(input_file_info@filename(0), "r")
  ;   if (isfilevar(f, "plev") .and. isfilevar(f, "lat")) then
  ;     plev = read_data(0, "plev", field_type0)
  ;     lat = read_data(0, "lat", field_type0)
  ;     if (dimsizes(plev) .eq. 1) then
  ;       lev = plev(0)
  ;     else
  ;       error_msg("f", diag_script, "", "ambiguous pressure levels given, " + \
  ;                 "only one supported")
  ;     end if
  ;     if (dimsizes(lat) .gt. 1) then
  ;       lat_range = (/min(lat), max(lat)/)
  ;     else
  ;       lat_range = read_data(0, "lat_bnds", field_type0)
  ;     end if
  ;   else
  ;     if (isatt(f, "plev") .and. \
  ;         isatt(f, "lat_range_0") .and. \
  ;         isatt(f, "lat_range_1")) then
  ;       lev = f@plev
  ;       lat_range = (/f@lat_range_0, f@lat_range_1/)
  ;     else
  ;       error_msg("f", diag_script, "", "cannot read plev and latrange")
  ;     end if
  ;   end if
  ;   log_info("Retrieved pressure level " + lev + " and latitude range " + \
  ;            lat_range(0) + " to " + lat_range(1) + " from preprocessed data")
  ; end if

  ; ---------------------------------------------------------------------------
  ; Read and preprocess data
  ; ---------------------------------------------------------------------------

  ; Iterate over datasets
  ; do i = 0, DIM_DAT - 1
  ;   idat = idx_datasets(i)
  ;   log_info(input_file_info@dataset(idat))

  ;   Read data
  ;   A0 = read_data(idat, var0, field_type0)

  ;   Average over ensemble members if desired
  ;   if (avgens) then
  ;     ens = ind(input_file_info@dataset(idat) .eq. input_file_info@dataset)
  ;     rank = dimsizes(dimsizes(A0))

  ;     Check if there is more than one ensemble member
  ;     if (dimsizes(ens) .gt. 1) then
  ;       dims = getVarDimNames(A0)
  ;       if (rank .eq. 4) then
  ;         tmpA0 = new((/dimsizes(ens), dimsizes(A0&$dims(0)$), \
  ;                     dimsizes(A0&$dims(1)$), dimsizes(A0&$dims(2)$), \
  ;                     dimsizes(A0&$dims(3)$)/), typeof(A0))
  ;         copy_VarCoords(A0, tmpA0(0, :, :, :, :))
  ;         tmpA0(0, :, :, :, :) = (/A0/)
  ;       else if (rank .eq. 3) then
  ;         tmpA0 = new((/dimsizes(ens), dimsizes(A0&$dims(0)$), \
  ;                     dimsizes(A0&$dims(1)$), dimsizes(A0&$dims(2)$)/), \
  ;                     typeof(A0))
  ;         copy_VarCoords(A0, tmpA0(0, :, :, :))
  ;         tmpA0(0, :, :, :) = (/A0/)
  ;       else if (rank .eq. 2) then
  ;         tmpA0 = new((/dimsizes(ens), dimsizes(A0&$dims(0)$), \
  ;                     dimsizes(A0&$dims(1)$)/), typeof(A0))
  ;         copy_VarCoords(A0, tmpA0(0, :, :))
  ;         tmpA0(0, :, :) = (/A0/)
  ;       else if (rank .eq. 1) then
  ;         tmpA0 = new((/dimsizes(ens), dimsizes(A0&$dims(0)$)/), typeof(A0))
  ;         copy_VarCoords(A0, tmpA0(0, :))
  ;         tmpA0(0, :) = (/A0/)
  ;       else
  ;         error_msg("f", diag_script, "", "unsupported rank (" + rank + \
  ;                   ") of variable " + var0)
  ;       end if
  ;       end if
  ;       end if
  ;       end if
  ;       delete(A0)

  ;       Average over ensemble members
  ;       tmpA0!0 = "ensemble_members"
  ;       do iens = 1, dimsizes(ens) - 1
  ;         tmp_data = read_data(idat + iens, var0, field_type0)
  ;         if (rank .eq. 4) then
  ;           tmpA0(iens, :, :, :, :) = (/tmp_data/)
  ;         else if (rank .eq. 3) then
  ;           tmpA0(iens, :, :, :) = (/tmp_data/)
  ;         else if (rank .eq. 2) then
  ;           tmpA0(iens, :, :) = (/tmp_data/)
  ;         else if (rank .eq. 1) then
  ;           tmpA0(iens, :) = (/tmp_data/)
  ;         end if
  ;         end if
  ;         end if
  ;         end if
  ;         delete(tmp_data)
  ;       end do
  ;       A0 = dim_avg_n_Wrap(tmpA0, 0)
  ;       delete(tmpA0)
  ;     end if
  ;     delete(ens)
  ;   end if

  ;   -------------------------------------------------------------------------
  ;   Process spatial data
  ;   -------------------------------------------------------------------------

  ;   Process mmstf (SH Hadley cell boundary)
  ;   if (var0 .eq. "mmstf") then
  ;     tmp = A0
  ;     dummy1 = new(dimsizes(A0&time), typeof(A0&lat))

  ;     1st degree polynomial fit for all times
  ;     do tt = 0, dimsizes(A0&time) - 1
  ;       idxmin = minind(tmp(tt, :))
  ;       idxmax = maxind(tmp(tt, :))
  ;       tmpp = tmp(tt, idxmin:idxmax)
  ;       idx1 = min(ind(tmpp .ge. 0.0))
  ;       idx2 = max(ind(tmpp .le. 0.0))
  ;       if (idx2 .gt. (idx1 - 1)) then
  ;         idx2 = idx1 - 1
  ;       end if
  ;       scal_x = tmpp(idx2:idx1)
  ;       scal_y = tmpp&lat(idx2:idx1)
  ;       pf = lspoly(scal_x, scal_y, 1, 2)
  ;       yfit = pf(0) + pf(1) * scal_x
  ;       dummy1(tt) = yfit(1)
  ;       delete([/pf, idx1, idx2, idxmin, idxmax, yfit, tmpp, scal_x, scal_y/])
  ;     end do
  ;     dummy1!0 = "time"
  ;     dummy1&time = A0&time
  ;     variable_info@units = "degrees"
  ;     delete(tmp)

  ;   Other variables (no operations needed, done in preprocessing)
  ;   else
  ;       dummy1 = A0
  ;   end if
  ;   end if
  ;   delete(A0)

  ;   Convert units of tro3
  ;   if (var0 .eq. "tro3") then
  ;     dummy1 = dummy1 * 10 ^ -3
  ;     variable_info@units = "ppmv"
  ;   end if

  ;   -------------------------------------------------------------------------
  ;   Process temporal data
  ;   -------------------------------------------------------------------------

  ;   if (season .ne. "yearly") then
  ;     tmp = time_operations(dummy1, \
  ;                           toint(input_file_info@start_year(idat)), \
  ;                           toint(input_file_info@end_year(idat)), \
  ;                           "extract", season, True)
  ;     dummy2 = time_operations(tmp, \
  ;                              toint(input_file_info@start_year(idat)), \
  ;                              toint(input_file_info@end_year(idat)), \
  ;                              "average", "yearly", True)
  ;     delete(tmp)
  ;   else
  ;     dummy2 = time_operations(dummy1, \
  ;                              toint(input_file_info@start_year(idat)), \
  ;                              toint(input_file_info@end_year(idat)), \
  ;                              "average", season, True)
  ;   end if

  ;   Trend
  ;   ttmp = ispan(toint(input_file_info@start_year(idat)), \
  ;                toint(input_file_info@end_year(idat)), 1)
  ;   dummy3 = dummy2 - dim_avg_n(dummy2, 0)
  ;   rc = regline(ttmp, dummy3)
  ;   if (var0 .ne. "tpp") then
  ;     val_trend(0, i) = tofloat(rc) * 10
  ;     val_trend(1, i) = tofloat(rc@rstd) * 10
  ;   else
  ;     val_trend(0, i) = tofloat(rc)
  ;     val_trend(1, i) = tofloat(rc@rstd)
  ;   end if

  ;   Mean
  ;   val_mean(0, i) = tofloat(avg(dummy2))
  ;   val_mean(1, i) = tofloat(stddev(dummy2) / sqrt(dimsizes(dummy2)))
  ;   delete([/dummy1, dummy2, dummy3/])
  ;   delete([/ttmp, rc/])
  ; end do
