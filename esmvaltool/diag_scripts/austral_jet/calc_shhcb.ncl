; #############################################################################
; austral_jet/calc_shhcb.ncl
; #############################################################################
; Author: Manuel Schlund (DLR, Germany)
; #############################################################################
;
; Description:
;     Simple script to calculate and save the Hadley cell boundary in the
;     southern hemisphere for the use in other diagnostics (e.g. MLR models).
;
; Required diag_script_info attributes:
;     calc_type : Calculation type (mean/trend)
;     tag : Tag for the diagnostic.
;     lev : Pressure level.
;     lat_range : Latitude range.
;     season : Season which is considered.
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

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"


begin

  enter_msg(DIAG_SCRIPT, "")

  ; Check variables
  VARS = metadata_att_as_array(variable_info, "short_name")
  if (.not. any(VARS .eq. "va")) then
    error_msg("f", DIAG_SCRIPT, "", "variable 'va' not given")
  end if
  if (.not. any(VARS .eq. "ps")) then
    error_msg("f", DIAG_SCRIPT, "", "variable 'ps' not given")
  end if

  ; Input data
  VA_INFO = select_metadata_by_name(input_file_info, "va")
  PS_INFO = select_metadata_by_name(input_file_info, "ps")

  ; Check if datasets are equal
  va_datasets = metadata_att_as_array(VA_INFO, "dataset")
  ps_datasets = metadata_att_as_array(PS_INFO, "dataset")
  if (.not. all(va_datasets .eq. ps_datasets)) then
    error_msg("f", DIAG_SCRIPT, "", "va and ps variables do not contain " + \
              "the same datasets")
  end if
  DATASETS = va_datasets
  DIM_DAT = dimsizes(DATASETS)

  ; Required attributes
  req_atts = (/"calc_type", "tag", "season", "lev", "lat_range"/)
  exit_if_missing_atts(diag_script_info, req_atts)
  req_var_atts = (/"var_type"/)
  do idat = 0, DIM_DAT - 1
    exit_if_missing_atts(VA_INFO[idat], req_var_atts)
    exit_if_missing_atts(PS_INFO[idat], req_var_atts)
  end do
  calc_type = diag_script_info@calc_type
  season = diag_script_info@season

  ; Create work directory
  system("mkdir -p " + config_user_info@work_dir)

end

begin

  ; Iterate over datasets
  do idat = 0, DIM_DAT - 1
    log_info(DATASETS(idat))
    info := VA_INFO[idat]
    va = read_data(info)
    ps = read_data(PS_INFO[idat])
    if (dimsizes(va&lat) .ne. dimsizes(ps&lat)) then
      tmp = ps({time|:}, {lat|va&lat}, {lon|:})
      delete(ps)
      ps = tmp
      delete(tmp)  ; ps := ps(...) causes memory leaks
    end if

    ; Preprocess plev (see documentation of NCL's zonal_mpsi_Wrap)
    pr = va&plev
    p_idx = ind(pr .gt. 500.0 .and. pr .le. 100500.0)
    pr := pr(p_idx)
    vt2b = va(:, p_idx, :, :)
    if (pr(0) .eq. max(pr)) then
      pr := pr(::-1)
      tmp = vt2b({time|:}, {plev|::-1}, {lat|:}, {lon|:})
      delete(vt2b)
      vt2b = tmp
      delete(tmp)  ; vt2b := vt2b(...) causes memory leaks
    end if

    ; Calculate zonal mean meridional stream function
    mmstf = zonal_mpsi_Wrap(vt2b, va&lat, pr, ps)
    mmstf!0 = va!0
    mmstf!1 = va!1
    mmstf!2 = va!2
    mmstf&time = va&time
    mmstf&plev = pr
    mmstf&lat = va&lat
    delete([/va, ps, pr, p_idx, vt2b/])

    ; Calculate southern hemisphere Hadley cell boundary
    lev = diag_script_info@lev
    lat_range = diag_script_info@lat_range
    mmstf := mmstf({time|:}, {plev|lev}, {lat|lat_range(0):lat_range(1)})
    outvar = new(dimsizes(mmstf&time), typeof(mmstf&lat))

    ; 1st degree polynomial fit for all times
    do tt = 0, dimsizes(mmstf&time) - 1
      idx_min = minind(mmstf(tt, :))
      idx_max = maxind(mmstf(tt, :))
      tmp = mmstf(tt, idx_min:idx_max)
      idx_1 = min(ind(tmp .ge. 0.0))
      idx_2 = max(ind(tmp .le. 0.0))
      if (ismissing(idx_1)) then
        idx_1 = 1
      end if
      if (ismissing(idx_2)) then
        idx_2 = dimsizes(tmp) - 1
      end if
      if (idx_2 .gt. (idx_1 - 1)) then
        idx_2 = idx_1 - 1
      end if
      scal_x = tmp(idx_2:idx_1)
      scal_y = tmp&lat(idx_2:idx_1)
      pf = lspoly(scal_x, scal_y, 1, 2)
      y_fit = pf(0) + pf(1) * scal_x
      outvar(tt) = y_fit(1)
      delete([/pf, idx_1, idx_2, idx_min, idx_max, y_fit, tmp, scal_x, \
              scal_y/])
    end do
    outvar!0 = "time"
    outvar&time = mmstf&time

    ; Extract season
    if (season .ne. "yearly") then
      outvar := time_operations(outvar, info@start_year, info@end_year, \
                                "extract", season, True)
    end if

    ; Calculation type
    if (calc_type .eq. "mean") then
      outvar := avg(outvar)
      outvar@var = "shhcb"
      outvar@long_name = "Boundary of Hadley cell in southern hemisphere"
      outvar@units = "degrees_north"
    else if (calc_type .eq. "trend") then
      if (isatt(diag_script_info, "yearly_trend") .and. \
          diag_script_info@yearly_trend) then
        outvar := time_operations(outvar, info@start_year, info@end_year, \
                                  "average", "yearly", True)
        time = ispan(info@start_year, info@end_year, 1)
        units_suffix = " yr-1"
      else
        time = outvar&time
        units_suffix = " mon-1"
      end if
      rc = regline(time, outvar)
      outvar := rc
      outvar@var = "shhcb_trend"
      outvar@long_name = "Boundary of Hadley cell in southern hemisphere " + \
        "(trend)"
      outvar@units = "degrees_north" + units_suffix
      delete([/time, rc, units_suffix/])
    else
      error_msg("f", DIAG_SCRIPT, "", "calc_type '" + calc_type + \
                "' not supported")
    end if
    end if

    ; Set other attributes
    outvar!0 = "time"
    outvar&time = avg((/info@start_year, info@end_year/))
    outvar@diag_script = DIAG_SCRIPT
    outvar@dataset = DATASETS(idat)
    if (isatt(info, "ensemble")) then
      outvar@ensemble = info@ensemble
    end if
    outvar@tag = diag_script_info@tag
    outvar@var_type = info@var_type
    outvar@project = info@project
    filename = systemfunc("basename " + info@filename)
    outvar@filename = config_user_info@work_dir + filename + "." + \
      outvar@var + ".nc"

    ; Save file
    ncdf_outfile = ncdf_write(outvar, outvar@filename)
    delete([/mmstf, outvar, ncdf_outfile/])
  end do

  leave_msg(DIAG_SCRIPT, "")

end
