!WRF:MODEL_LAYER:DYNAMICS
!
MODULE module_avgflx_em

  USE module_bc
  USE module_model_constants
  USE module_wrf_error
  USE module_state_description

CONTAINS

!-------------------------------------------------------------------------------


  subroutine zero_avgflx(avgflx_rum,avgflx_rvm,avgflx_wwm, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte,           &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1, &
       & config_flags,                                                                           &
       & th_mean,thx_mean,thy_mean,thz_mean,thth_mean,ruth_tot_mean,rvth_tot_mean,wwth_tot_mean, &
       & q_mean,qx_mean,qy_mean,qz_mean,qq_mean,ruq_tot_mean,rvq_tot_mean,wwq_tot_mean,          &
       & u_mean,ux_mean,uy_mean,uz_mean,ruu_tot_mean,rvu_tot_mean,wwu_tot_mean,                  &
       & v_mean,vx_mean,vy_mean,vz_mean,ruv_tot_mean,rvv_tot_mean,wwv_tot_mean,                  &
       & om_mean,wd_mean,zwind_mean,wx_mean,wy_mean,wz_mean,ruw_tot_mean,rvw_tot_mean,www_tot_mean,  &
       & corr_uth,corr_vth,corr_dthdt,                                                           &
       & corr_uq,corr_vq,corr_dqdt,                                                              &
       & corr_uu,corr_vu,corr_dudt,                                                              &
       & corr_uv,corr_vv,corr_dvdt,                                                              &
       & corr_uw,corr_vw,corr_dwdt,                                                              &
       & sgs_UTH_mean,sgs_VTH_mean,sgs_WTH_mean,                                                 &
       & sgs_UQ_mean,sgs_VQ_mean,sgs_WQ_mean,                                                    &
       & sgs_UU_mean,sgs_WU_mean,                                                                &
       & sgs_UV_mean,sgs_VV_mean,sgs_WV_mean,                                                    &
       & sgs_UW_mean,sgs_VW_mean,sgs_WW_mean,                                                    &
       & t_tend_radlw_mean,t_tend_radsw_mean,t_tend_mp_mean,q_tend_mp_mean,                      &
       & rhod_mean,z_mean                                                                        )

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags

    REAL, OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(OUT) ::              &
       th_mean,thx_mean,thy_mean,thz_mean,thth_mean,ruth_tot_mean,rvth_tot_mean,wwth_tot_mean, &
       q_mean,qx_mean,qy_mean,qz_mean,qq_mean,ruq_tot_mean,rvq_tot_mean,wwq_tot_mean,          &
       u_mean,ux_mean,uy_mean,uz_mean,ruu_tot_mean,rvu_tot_mean,wwu_tot_mean,                  &
       v_mean,vx_mean,vy_mean,vz_mean,ruv_tot_mean,rvv_tot_mean,wwv_tot_mean,                  &
       om_mean,wd_mean,zwind_mean,wx_mean,wy_mean,wz_mean,ruw_tot_mean,rvw_tot_mean,www_tot_mean,  &
       corr_uth,corr_vth,corr_dthdt,                                                           &
       corr_uq,corr_vq,corr_dqdt,                                                              &
       corr_uu,corr_vu,corr_dudt,                                                              &
       corr_uv,corr_vv,corr_dvdt,                                                              &
       corr_uw,corr_vw,corr_dwdt,                                                              &
       sgs_UTH_mean,sgs_VTH_mean,sgs_WTH_mean,                                                 &
       sgs_UQ_mean,sgs_VQ_mean,sgs_WQ_mean,                                                    &
       sgs_UU_mean,sgs_WU_mean,                                                                &
       sgs_UV_mean,sgs_VV_mean,sgs_WV_mean,                                                    &
       sgs_UW_mean,sgs_VW_mean,sgs_WW_mean,                                                    &
       t_tend_radlw_mean,t_tend_radsw_mean,t_tend_mp_mean,q_tend_mp_mean,                      &
       rhod_mean,z_mean



    INTEGER :: i,j,k

    if (config_flags%do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = 0.
             avgflx_rvm(i,k,j) = 0.
             avgflx_wwm(i,k,j) = 0.
          end DO
       end DO
    end DO
    endif

    if ((config_flags%do_avgflx_cugd .EQ. 1) .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = 0.
                avgflx_cfd1(i,k,j) = 0.
                avgflx_dfu1(i,k,j) = 0.
                avgflx_efu1(i,k,j) = 0.
                avgflx_dfd1(i,k,j) = 0.
                avgflx_efd1(i,k,j) = 0.
             end DO
          end DO
       end DO
    end if

    if (present(z_mean) .and. config_flags%output_avgfluxes) then
      DO j=jts,jte
         DO k=kts,kte
            DO i=its,ite
                if (config_flags%output_th_fluxes > 0) then
                    if (config_flags%output_thres_fluxes) then
                        th_mean(i,k,j) = 0.
                        thx_mean(i,k,j) = 0.
                        thy_mean(i,k,j) = 0.
                        thz_mean(i,k,j) = 0.
                        thth_mean(i,k,j) = 0.
                        ruth_tot_mean(i,k,j) = 0.
                        rvth_tot_mean(i,k,j) = 0.
                        wwth_tot_mean(i,k,j) = 0.
                        corr_uth(i,k,j) = 0.
                        corr_vth(i,k,j) = 0.
                        corr_dthdt(i,k,j) = 0.
                    endif
                    if (config_flags%output_thsgs_fluxes) then
                        sgs_UTH_mean(i,k,j) = 0.
                        sgs_VTH_mean(i,k,j) = 0.
                        sgs_WTH_mean(i,k,j) = 0.
                    endif
                    t_tend_radlw_mean(i,k,j) = 0.
                    t_tend_radsw_mean(i,k,j) = 0.
                    t_tend_mp_mean(i,k,j) = 0.
                endif
                if (config_flags%output_q_fluxes > 0) then
                    if (config_flags%output_qres_fluxes) then
                        q_mean(i,k,j) = 0.
                        qx_mean(i,k,j) = 0.
                        qy_mean(i,k,j) = 0.
                        qz_mean(i,k,j) = 0.
                        qq_mean(i,k,j) = 0.
                        ruq_tot_mean(i,k,j) = 0.
                        rvq_tot_mean(i,k,j) = 0.
                        wwq_tot_mean(i,k,j) = 0.
                        corr_uq(i,k,j) = 0.
                        corr_vq(i,k,j) = 0.
                        corr_dqdt(i,k,j) = 0.
                    endif
                    if (config_flags%output_qsgs_fluxes) then
                        sgs_UQ_mean(i,k,j) = 0.
                        sgs_VQ_mean(i,k,j) = 0.
                        sgs_WQ_mean(i,k,j) = 0.
                    endif
                    q_tend_mp_mean(i,k,j) = 0.
                endif
                if (config_flags%output_u_fluxes > 0) then
                    if (config_flags%output_ures_fluxes) then
                        u_mean(i,k,j) = 0.
                        ux_mean(i,k,j) = 0.
                        uy_mean(i,k,j) = 0.
                        uz_mean(i,k,j) = 0.
                        ruu_tot_mean(i,k,j) = 0.
                        rvu_tot_mean(i,k,j) = 0.
                        wwu_tot_mean(i,k,j) = 0.
                        corr_uu(i,k,j) = 0.
                        corr_vu(i,k,j) = 0.
                        corr_dudt(i,k,j) = 0.
                    endif
                    if (config_flags%output_usgs_fluxes) then
                        sgs_UU_mean(i,k,j) = 0.
                        sgs_UV_mean(i,k,j) = 0.
                        sgs_WU_mean(i,k,j) = 0.
                    endif
                endif
                if (config_flags%output_v_fluxes > 0) then
                    if (config_flags%output_vres_fluxes) then
                        v_mean(i,k,j) = 0.
                        vx_mean(i,k,j) = 0.
                        vy_mean(i,k,j) = 0.
                        vz_mean(i,k,j) = 0.
                        ruv_tot_mean(i,k,j) = 0.
                        rvv_tot_mean(i,k,j) = 0.
                        wwv_tot_mean(i,k,j) = 0.
                        corr_uv(i,k,j) = 0.
                        corr_vv(i,k,j) = 0.
                        corr_dvdt(i,k,j) = 0.
                    endif
                    if (config_flags%output_vsgs_fluxes) then
                        sgs_UV_mean(i,k,j) = 0.
                        sgs_VV_mean(i,k,j) = 0.
                        sgs_WV_mean(i,k,j) = 0.
                    endif
                endif
                if (config_flags%output_w_fluxes > 0) then
                    if (config_flags%output_wres_fluxes) then
                        om_mean(i,k,j) = 0.
                        wd_mean(i,k,j) = 0.
                        zwind_mean(i,k,j) = 0.
                        wx_mean(i,k,j) = 0.
                        wy_mean(i,k,j) = 0.
                        wz_mean(i,k,j) = 0.
                        ruw_tot_mean(i,k,j) = 0.
                        rvw_tot_mean(i,k,j) = 0.
                        www_tot_mean(i,k,j) = 0.
                        corr_uw(i,k,j) = 0.
                        corr_vw(i,k,j) = 0.
                        corr_dwdt(i,k,j) = 0.
                    endif
                    if (config_flags%output_wsgs_fluxes) then
                        sgs_UW_mean(i,k,j) = 0.
                        sgs_VW_mean(i,k,j) = 0.
                        sgs_WW_mean(i,k,j) = 0.
                    endif
                endif

                rhod_mean(i,k,j) = 0.
                z_mean(i,k,j) = 0.

            end DO
         end DO
       end DO
    endif

    return
  end subroutine zero_avgflx

  subroutine upd_avgflx(avgflx_count,avgflx_rum,avgflx_rvm,avgflx_wwm, &
       &   ru_m, rv_m, ww_m, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte, do_cu,    &
       & cfu1,cfd1,dfu1,efu1,dfd1,efd1,          &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1,   &
       & config_flags,                                                              &
       & TH_MEAN,THX_MEAN,THY_MEAN,THZ_MEAN,THTH_MEAN,RUTH_TOT_MEAN,RVTH_TOT_MEAN,WWTH_TOT_MEAN, &
       & Q_MEAN,QX_MEAN,QY_MEAN,QZ_MEAN,QQ_MEAN,RUQ_TOT_MEAN,RVQ_TOT_MEAN,WWQ_TOT_MEAN,          &
       & U_MEAN,UX_MEAN,UY_MEAN,UZ_MEAN,RUU_TOT_MEAN,RVU_TOT_MEAN,WWU_TOT_MEAN,                  &
       & V_MEAN,VX_MEAN,VY_MEAN,VZ_MEAN,RUV_TOT_MEAN,RVV_TOT_MEAN,WWV_TOT_MEAN,                  &
       & OM_MEAN,WD_MEAN,ZWIND_MEAN,WX_MEAN,WY_MEAN,WZ_MEAN,RUW_TOT_MEAN,RVW_TOT_MEAN,WWW_TOT_MEAN,  &
       & CORR_UTH_MEAN,CORR_VTH_MEAN,CORR_DTHDT_MEAN,                                            &
       & CORR_UQ_MEAN,CORR_VQ_MEAN,CORR_DQDT_MEAN,                                               &
       & CORR_UU_MEAN,CORR_VU_MEAN,CORR_DUDT_MEAN,                                               &
       & CORR_UV_MEAN,CORR_VV_MEAN,CORR_DVDT_MEAN,                                               &
       & CORR_UW_MEAN,CORR_VW_MEAN,CORR_DWDT_MEAN,                                               &
       & SGS_UTH_MEAN,SGS_VTH_MEAN,SGS_WTH_MEAN,                                                 &
       & SGS_UQ_MEAN,SGS_VQ_MEAN,SGS_WQ_MEAN,                                                    &
       & SGS_UU_MEAN,SGS_WU_MEAN,                                                                &
       & SGS_UV_MEAN,SGS_VV_MEAN,SGS_WV_MEAN,                                                    &
       & SGS_UW_MEAN,SGS_VW_MEAN,SGS_WW_MEAN,                                                    &
       & T_TEND_RADLW_MEAN,T_TEND_RADSW_MEAN,T_TEND_MP_MEAN,Q_TEND_MP_MEAN,                      &
       & RHOD_MEAN,Z_MEAN,                                                                       &
       & ruth_tot, rvth_tot, wwth_tot, ruq_tot, rvq_tot, wwq_tot,                                &
       & ruu_tot, rvu_tot, wwu_tot, ruv_tot, rvv_tot, wwv_tot, ruw_tot, rvw_tot, www_tot,        &
       & SGS_UTH, SGS_VTH, SGS_WTH, SGS_UQ, SGS_VQ, SGS_WQ,                                      &
       & SGS_UU, SGS_WU, SGS_UV, SGS_VV, SGS_WV, SGS_UW, SGS_VW, SGS_WW,                         &
       & t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp,                                      &
       & U, V, W, ru, rv, ww, thm, Q,                                                            &
       & ph_1, ph_2, phb, zx, zy,                                                                &
       & dnw, fnm, fnp, mut, c1h, c2h, c1f, c2f, cf1, cf2, cf3, cfn, cfn1,                       &
       & last_timestep, dt                                                                       )
    

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    INTEGER , INTENT(IN)        :: avgflx_count
    LOGICAL, INTENT(IN) :: do_cu
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: ru_m, &
         rv_m, &
         ww_m

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(IN) ::    &
         cfu1,cfd1,dfu1,efu1,dfd1,efd1
    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1


    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::            &
       TH_MEAN,THX_MEAN,THY_MEAN,THZ_MEAN,THTH_MEAN,RUTH_TOT_MEAN,RVTH_TOT_MEAN,WWTH_TOT_MEAN, &
       Q_MEAN,QX_MEAN,QY_MEAN,QZ_MEAN,QQ_MEAN,RUQ_TOT_MEAN,RVQ_TOT_MEAN,WWQ_TOT_MEAN,          &
       U_MEAN,UX_MEAN,UY_MEAN,UZ_MEAN,RUU_TOT_MEAN,RVU_TOT_MEAN,WWU_TOT_MEAN,                  &
       V_MEAN,VX_MEAN,VY_MEAN,VZ_MEAN,RUV_TOT_MEAN,RVV_TOT_MEAN,WWV_TOT_MEAN,                  &
       OM_MEAN,WD_MEAN,ZWIND_MEAN,WX_MEAN,WY_MEAN,WZ_MEAN,RUW_TOT_MEAN,RVW_TOT_MEAN,WWW_TOT_MEAN,  &
       CORR_UTH_MEAN,CORR_VTH_MEAN,CORR_DTHDT_MEAN,                                                           &
       CORR_UQ_MEAN,CORR_VQ_MEAN,CORR_DQDT_MEAN,                                                              &
       CORR_UU_MEAN,CORR_VU_MEAN,CORR_DUDT_MEAN,                                                              &
       CORR_UV_MEAN,CORR_VV_MEAN,CORR_DVDT_MEAN,                                                              &
       CORR_UW_MEAN,CORR_VW_MEAN,CORR_DWDT_MEAN,                                                              &
       SGS_UTH_MEAN,SGS_VTH_MEAN,SGS_WTH_MEAN,                                                 &
       SGS_UQ_MEAN,SGS_VQ_MEAN,SGS_WQ_MEAN,                                                    &
       SGS_UU_MEAN,SGS_WU_MEAN,                                                                &
       SGS_UV_MEAN,SGS_VV_MEAN,SGS_WV_MEAN,                                                    &
       SGS_UW_MEAN,SGS_VW_MEAN,SGS_WW_MEAN,                                                    &
       T_TEND_RADLW_MEAN,T_TEND_RADSW_MEAN,T_TEND_MP_MEAN,Q_TEND_MP_MEAN,                      &
       RHOD_MEAN,Z_MEAN

    REAL, OPTIONAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
         U, V, W, ru, rv, ww,  &
         thm, Q, ph_1, ph_2, phb, zx, zy,  &
         ruth_tot, rvth_tot, wwth_tot,     &
         ruq_tot, rvq_tot, wwq_tot,        &
         ruu_tot, rvu_tot, wwu_tot,        &
         ruv_tot, rvv_tot, wwv_tot,        &
         ruw_tot, rvw_tot, www_tot,        &
         t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp, &
         SGS_UTH, SGS_VTH, SGS_WTH, SGS_UQ, SGS_VQ, SGS_WQ, &
         SGS_UU, SGS_WU, SGS_UV, SGS_VV, SGS_WV, SGS_UW, SGS_VW, SGS_WW

    REAL , DIMENSION( kms:kme ) , OPTIONAL, INTENT(IN   ) :: dnw, fnm, fnp, c1h, c2h, c1f, c2f
    REAL , DIMENSION( ims:ime, jms:jme ), OPTIONAL, INTENT(IN   ) :: mut
    REAL, OPTIONAL, INTENT(IN) :: dt, cf1, cf2, cf3, cfn, cfn1
    LOGICAL, OPTIONAL, INTENT(IN) :: last_timestep


    REAL,    DIMENSION( its-1:ite , kts:kte , jts-1:jte ) ::    rho_d, th, thd
    REAL,    DIMENSION( its-1:ite+1 , kts:kte , jts-1:jte+1 ) ::    dzdt, u8w, v8w, rho8w

    INTEGER :: i,j,k
    REAL :: local_count, rhox, rhoy, rhoz, rhoxy, rhoxz, rhoyz,                &
            rho_mean, rhox_mean, rhoy_mean, rhoz_mean, rhoxy_mean, rhoxz_mean, rhoyz_mean, &
            thetax, thetay, thetaz, qx, qy, qz, ux, uy, uz, vx, vy, vz, wx, wy, wz, qfac,     &
            ru_d, rv_d, ww_d,               &
            zx_s, zy_s, u_s, v_s, rho_s, dzdt_s, &
            corr_uth, corr_vth, corr_dthdt, &
            corr_uq, corr_vq, corr_dqdt,    &
            corr_uu, corr_vu, corr_dudt, &
            corr_uv, corr_vv, corr_dvdt, &
            corr_uw, corr_vw, corr_dwdt, &
            wdiag, muz, mu, dphidn

    local_count = real(avgflx_count)

    if (present(dt) .and. config_flags%output_avgfluxes) then

    DO j=jts-1,jte
    DO k=kts,kte
    DO i=its-1,ite
      !calculate dry density
      mu = c1h(k)*mut(i,j)+c2h(k)
      dphidn = ((phb(i,k+1,j)+ph_2(i,k+1,j)) - (phb(i,k,j)+ph_2(i,k,j)))/dnw(k)
      rho_d(i,k,j) = mu/(-dphidn)
      RHOD_MEAN(i,k,j) = (local_count*RHOD_MEAN(i,k,j)  + rho_d(i,k,j)  )/(local_count+1.)
      IF (config_flags%output_thres_fluxes) THEN
          IF ( config_flags%use_theta_m .EQ. 1 ) THEN
            thd(i,k,j) = (thm(i,k,j) + t0)/(1 + rvovrd*Q(i,k,j)) - t0
            IF ( config_flags%output_dry_theta_fluxes) THEN
              th(i,k,j) = thd(i,k,j)
            ELSE
              th(i,k,j) = thm(i,k,j)
            ENDIF
          ELSE
            thd(i,k,j) = thm(i,k,j)
            th(i,k,j) = thm(i,k,j)
          ENDIF
      ENDIF
    ENDDO
    ENDDO
    ENDDO

    IF (config_flags%output_res_fluxes) THEN
        DO j=jts-1,jte+1
        DO k=kts,kte
        DO i=its-1,ite+1
            dzdt(i,k,j) = (ph_2(i,k,j) - ph_1(i,k,j))/(g*dt)
            call stagger_z(u8w(i,k,j), U(i,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )
            call stagger_z(v8w(i,k,j), V(i,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )
            call stagger_z(rho8w(i,k,j), rho_d(i,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )
        ENDDO
        ENDDO
        ENDDO
    ENDIF

    DO j=jts,jte
      DO k=kts,kte
        DO i=its,ite

          Z_MEAN(i,k,j) = (local_count*Z_MEAN(i,k,j)  + (phb(i,k,j)+ph_2(i,k,j))/g  )/(local_count+1.)

          IF (config_flags%output_res_fluxes) THEN
            !stagger density
            if (config_flags%hesselberg_avg) then
              rhox = 0.5*(rho_d(i,k,j)+rho_d(i-1,k,j))
              rhoy = 0.5*(rho_d(i,k,j)+rho_d(i,k,j-1))
              rhoxy = 0.25*(rho_d(i,k,j)+rho_d(i,k,j-1)+rho_d(i-1,k,j)+rho_d(i-1,k,j-1))
              rhoz = rho8w(i,k,j)
              call stagger_z(rhoxz, 0.5*(rho_d(i,:,j)+rho_d(i-1,:,j)),  &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                             i, j, k, kms, kme, kde              )
              call stagger_z(rhoyz, 0.5*(rho_d(i,:,j)+rho_d(i,:,j-1)),  &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                             i, j, k, kms, kme, kde              )
            else
              rhox = 1
              rhoy = 1
              rhoxy = 1
              rhoz = 1
              rhoxz = 1
              rhoyz = 1
            endif


            !determine staggered variables from total fluxes
            IF (config_flags%output_thres_fluxes) THEN
                call get_staggered_from_fluxes(thetax, thetay, thetaz,      &
                                               ruth_tot, rvth_tot, wwth_tot,&
                                               ru(i,k,j), rv(i,k,j), ww(i,k,j), th, &
                                               cf1, cf2, cf3, cfn, cfn1, .false.,   &
                                               i, j, k,                             &
                                               ids, ide, jds, jde, kds, kde,        &
                                               ims, ime, jms, jme, kms, kme         )

                if ((abs(thetax) .gt. 50) .or. (abs(thetay) .gt. 50) .or. (abs(thetaz) .gt. 50)) then
                !TODO: also do for other variables!
                    WRITE ( wrf_err_message , FMT=* ) "Found large thetax/y/z:", thetax, thetay, thetaz, "at i,j,k:",i,",",j,",",k
                    CALL wrf_debug ( 0 , wrf_err_message )
                endif

            ENDIF
            !staggered q, different vertical velocity needed
            IF (config_flags%output_qres_fluxes) THEN
                call get_staggered_from_fluxes(qx, qy, qz,                 &
                                               ruq_tot, rvq_tot, wwq_tot,  &
                                               ru_m(i,k,j), rv_m(i,k,j), ww_m(i,k,j), Q, &
                                                cf1, cf2, cf3, cfn, cfn1, .false.,       &
                                               i, j, k,                                  &
                                               ids, ide, jds, jde, kds, kde,             &
                                               ims, ime, jms, jme, kms, kme              )
            ENDIF
            IF (config_flags%output_ures_fluxes) THEN
                !(de)stagger in x direction
                ru_d = 0.5*(ru(i,k,j)+ru(i+1,k,j))
                rv_d = 0.5*(rv(i-1,k,j)+rv(i,k,j))
                ww_d = 0.5*(ww(i-1,k,j)+ww(i,k,j))
                call get_staggered_from_fluxes(ux, uy, uz,                 &
                                               ruu_tot, rvu_tot, wwu_tot,  &
                                               ru_d, rv_d, ww_d, U,        &
                                               cf1, cf2, cf3, cfn, cfn1, .false., &
                                               i, j, k,                           &
                                               ids, ide, jds, jde, kds, kde,      &
                                               ims, ime, jms, jme, kms, kme       )
            ENDIF
            IF (config_flags%output_vres_fluxes) THEN
                !(de)stagger in y direction
                ru_d = 0.5*(ru(i,k,j-1)+ru(i,k,j))
                rv_d = 0.5*(rv(i,k,j+1)+rv(i,k,j))
                ww_d = 0.5*(ww(i,k,j-1)+ww(i,k,j))
                call get_staggered_from_fluxes(vx, vy, vz,                 &
                                               ruv_tot, rvv_tot, wwv_tot,  &
                                               ru_d, rv_d, ww_d, V,        &
                                               cf1, cf2, cf3, cfn, cfn1, .false., &
                                               i, j, k,                           &
                                               ids, ide, jds, jde, kds, kde,      &
                                               ims, ime, jms, jme, kms, kme       )
            ENDIF
            IF (config_flags%output_wres_fluxes) THEN
                IF (k .eq. kts) THEN
                    wx = 0
                    wy = 0
                    wz = 0
                ELSE
                    IF (k .eq. kte) THEN
                        ru_d = (2.-fnm(k-1))*ru(i,k-1,j)-fnp(k-1)*ru(i,k-2,j)
                        rv_d = (2.-fnm(k-1))*rv(i,k-1,j)-fnp(k-1)*rv(i,k-2,j)
                        ww_d = ww(i,k-1,j)
                    ELSE
                        ru_d = fnm(k)*ru(i,k,j)+fnp(k)*ru(i,k-1,j)
                        rv_d = fnm(k)*rv(i,k,j)+fnp(k)*rv(i,k-1,j)
                        ww_d = 0.5*(ww(i,k,j)+ww(i,k+1,j))
                    ENDIF

                    call get_staggered_from_fluxes(wx, wy, wz,                 &
                                                   ruw_tot, rvw_tot, www_tot,  &
                                                   ru_d, rv_d, ww_d, W,        &
                                                   cf1, cf2, cf3, cfn, cfn1, .true.,  &
                                                   i, j, k,                           &
                                                   ids, ide, jds, jde, kds, kde,      &
                                                   ims, ime, jms, jme, kms, kme       )
                ENDIF
            ENDIF

            !compute cartesian corrections

            !destagger to masspoints
            zx_s = 0.5*(zx(i,k,j) + zx(i+1,k,j))
            zy_s = 0.5*(zy(i,k,j) + zy(i,k,j+1))
            u_s = 0.5*(u8w(i,k,j) + u8w(i+1,k,j))
            v_s = 0.5*(v8w(i,k,j) + v8w(i,k,j+1))
            dzdt_s = dzdt(i,k,j)
            rho_s = rho8w(i,k,j)

            wdiag = dzdt_s + zx_s*u_s + zy_s*v_s + ww(i,k,j)/(-rho_s*g)
            muz = c1f(k)*mut(i,j)+c2f(k)

            IF (config_flags%output_thres_fluxes) THEN
                corr_uth = rho_s*zx_s*u_s*thetaz
                corr_vth = rho_s*zy_s*v_s*thetaz
                corr_dthdt = rho_s*dzdt_s*thetaz
            ENDIF
            IF (config_flags%output_qres_fluxes) THEN
                corr_uq = rho_s*zx_s*u_s*qz
                corr_vq = rho_s*zy_s*v_s*qz
                corr_dqdt = rho_s*dzdt_s*qz
            ENDIF
            IF (config_flags%output_ures_fluxes) THEN
                zx_s = zx(i,k,j)
                zy_s = 0.25*(zy(i,k,j) + zy(i,k,j+1) + zy(i-1,k,j) + zy(i-1,k,j+1))
                u_s = u8w(i,k,j)
                v_s = 0.25*(v8w(i,k,j) + v8w(i,k,j+1) + v8w(i-1,k,j) + v8w(i-1,k,j+1))
                dzdt_s = 0.5*(dzdt(i,k,j) + dzdt(i-1,k,j))
                rho_s = 0.5*(rho8w(i,k,j) + rho8w(i-1,k,j))

                corr_uu = rho_s*zx_s*u_s*uz
                corr_vu = rho_s*zy_s*v_s*uz
                corr_dudt = rho_s*dzdt_s*uz
            ENDIF
            IF (config_flags%output_vres_fluxes) THEN
                zx_s = 0.25*(zx(i,k,j) + zx(i+1,k,j) + zx(i,k,j-1) + zx(i+1,k,j-1))
                zy_s = zy(i,k,j)
                u_s = 0.25*(u8w(i,k,j) + u8w(i+1,k,j) + u8w(i,k,j-1) + u8w(i+1,k,j-1))
                v_s = v8w(i,k,j)
                dzdt_s = 0.5*(dzdt(i,k,j-1) + dzdt(i,k,j-1))
                rho_s = 0.5*(rho8w(i,k,j-1) + rho8w(i,k,j-1))

                corr_uv = rho_s*zx_s*u_s*vz
                corr_vv = rho_s*zy_s*v_s*vz
                corr_dvdt = rho_s*dzdt_s*vz
            ENDIF
            IF (config_flags%output_wres_fluxes) THEN
                IF (k .lt. kte) THEN
                    !destagger in z and x/y direction
                    zx_s = 0.25*(zx(i,k,j) + zx(i+1,k,j) + zx(i,k+1,j) + zx(i+1,k+1,j))
                    zy_s = 0.25*(zy(i,k,j) + zy(i,k,j+1) + zy(i,k+1,j) + zy(i,k+1,j+1))
                    u_s = 0.5*(U(i,k,j) + U(i+1,k,j))
                    v_s = 0.5*(V(i,k,j) + V(i,k,j+1))
                    dzdt_s = 0.5*(dzdt(i,k,j) + dzdt(i,k+1,j))
                    rho_s = rho_d(i,k,j)

                    corr_uw = rho_s*zx_s*u_s*wz
                    corr_vw = rho_s*zy_s*v_s*wz
                    corr_dwdt = rho_s*dzdt_s*wz
                ELSE
                    corr_uw = 0
                    corr_vw = 0
                    corr_dwdt = 0
                ENDIF
            ENDIF

            if (last_timestep .and. config_flags%hesselberg_avg) then
              !normalize means by density
              rho_mean = RHOD_MEAN(i,k,j)
              rhox_mean = 0.5*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i-1,k,j))
              rhoy_mean = 0.5*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i,k,j-1))
              rhoxy_mean = 0.25*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i,k,j-1) + RHOD_MEAN(i-1,k,j) + RHOD_MEAN(i-1,k,j-1))

              call stagger_z(rhoz_mean, RHOD_MEAN(i,:,j), &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                             i, j, k, kms, kme, kde                   )
              call stagger_z(rhoxz_mean, 0.5*(RHOD_MEAN(i,:,j) + RHOD_MEAN(i-1,:,j)), &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                             i, j, k, kms, kme, kde                   )
              call stagger_z(rhoyz_mean, 0.5*(RHOD_MEAN(i,:,j) + RHOD_MEAN(i,:,j-1)), &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                             i, j, k, kms, kme, kde                   )

            ELSE
              rho_mean  = 1
              rhox_mean = 1
              rhoy_mean = 1
              rhoxy_mean = 1
              rhoz_mean = 1
              rhoxz_mean = 1
              rhoyz_mean = 1
            ENDIF

            !mean variables

            U_MEAN(i,k,j) = (local_count*U_MEAN(i,k,j) + U(i,k,j)*rhox  )/(local_count+1.)/rhox_mean
            V_MEAN(i,k,j) = (local_count*V_MEAN(i,k,j) + V(i,k,j)*rhoy  )/(local_count+1.)/rhoy_mean
            ZWIND_MEAN(i,k,j) = (local_count*ZWIND_MEAN(i,k,j) + W(i,k,j)*rhoz  )/(local_count+1.)/rhoz_mean
            OM_MEAN(i,k,j) = (local_count*OM_MEAN(i,k,j) + ww(i,k,j)/muz*rhoz )/(local_count+1.)/rhoz_mean
            WD_MEAN(i,k,j) = (local_count*WD_MEAN(i,k,j) + wdiag*rhoz )/(local_count+1.)/rhoz_mean

            IF (config_flags%output_thres_fluxes) THEN
                TH_MEAN(i,k,j) = (local_count*TH_MEAN(i,k,j) + thd(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
                THX_MEAN(i,k,j) = (local_count*THX_MEAN(i,k,j) + thetax*rhox )/(local_count+1.)/rhox_mean
                THY_MEAN(i,k,j) = (local_count*THY_MEAN(i,k,j) + thetay*rhoy )/(local_count+1.)/rhoy_mean
                THZ_MEAN(i,k,j) = (local_count*THZ_MEAN(i,k,j) + thetaz*rhoz )/(local_count+1.)/rhoz_mean
                THTH_MEAN(i,k,j) = (local_count*THTH_MEAN(i,k,j) + thd(i,k,j)**2 )/(local_count+1.)
                RUTH_TOT_MEAN(i,k,j)  = (local_count*RUTH_TOT_MEAN(i,k,j) + ruth_tot(i,k,j) )/(local_count+1.)
                RVTH_TOT_MEAN(i,k,j)  = (local_count*RVTH_TOT_MEAN(i,k,j) + rvth_tot(i,k,j) )/(local_count+1.)
                WWTH_TOT_MEAN(i,k,j)  = (local_count*WWTH_TOT_MEAN(i,k,j) + wwth_tot(i,k,j) )/(local_count+1.)
                CORR_UTH_MEAN(i,k,j)  = (local_count*CORR_UTH_MEAN(i,k,j) + corr_uth )/(local_count+1.)
                CORR_VTH_MEAN(i,k,j)  = (local_count*CORR_VTH_MEAN(i,k,j) + corr_vth )/(local_count+1.)
                CORR_DTHDT_MEAN(i,k,j)  = (local_count*CORR_DTHDT_MEAN(i,k,j) + corr_dthdt )/(local_count+1.)
            ENDIF
            IF (config_flags%output_qres_fluxes) THEN
                Q_MEAN(i,k,j) = (local_count*Q_MEAN(i,k,j) + Q(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
                QX_MEAN(i,k,j) = (local_count*QX_MEAN(i,k,j) + qx*rhox )/(local_count+1.)/rhox_mean
                QY_MEAN(i,k,j) = (local_count*QY_MEAN(i,k,j) + qy*rhoy )/(local_count+1.)/rhoy_mean
                QZ_MEAN(i,k,j) = (local_count*QZ_MEAN(i,k,j) + qz*rhoz )/(local_count+1.)/rhoz_mean
                QQ_MEAN(i,k,j) = (local_count*QQ_MEAN(i,k,j) + Q(i,k,j)*2 )/(local_count+1.)
                RUQ_TOT_MEAN(i,k,j)  = (local_count*RUQ_TOT_MEAN(i,k,j) + ruq_tot(i,k,j) )/(local_count+1.)
                RVQ_TOT_MEAN(i,k,j)  = (local_count*RVQ_TOT_MEAN(i,k,j) + rvq_tot(i,k,j) )/(local_count+1.)
                WWQ_TOT_MEAN(i,k,j)  = (local_count*WWQ_TOT_MEAN(i,k,j) + wwq_tot(i,k,j) )/(local_count+1.)
                CORR_UQ_MEAN(i,k,j)  = (local_count*CORR_UQ_MEAN(i,k,j) + corr_uq )/(local_count+1.)
                CORR_VQ_MEAN(i,k,j)  = (local_count*CORR_VQ_MEAN(i,k,j) + corr_vq )/(local_count+1.)
                CORR_DQDT_MEAN(i,k,j)  = (local_count*CORR_DQDT_MEAN(i,k,j) + corr_dqdt )/(local_count+1.)
            ENDIF

            IF (config_flags%output_ures_fluxes) THEN
                UX_MEAN(i,k,j) = (local_count*UX_MEAN(i,k,j) + ux*rho_d(i,k,j) )/(local_count+1.)/rho_mean
                UY_MEAN(i,k,j) = (local_count*UY_MEAN(i,k,j) + uy*rhoxy )/(local_count+1.)/rhoxy_mean
                UZ_MEAN(i,k,j) = (local_count*UZ_MEAN(i,k,j) + uz*rhoxz )/(local_count+1.)/rhoxz_mean
                RUU_TOT_MEAN(i,k,j)  = (local_count*RUU_TOT_MEAN(i,k,j) + ruu_tot(i,k,j) )/(local_count+1.)
                RVU_TOT_MEAN(i,k,j)  = (local_count*RVU_TOT_MEAN(i,k,j) + rvu_tot(i,k,j) )/(local_count+1.)
                WWU_TOT_MEAN(i,k,j)  = (local_count*WWU_TOT_MEAN(i,k,j) + wwu_tot(i,k,j) )/(local_count+1.)
                CORR_UU_MEAN(i,k,j)  = (local_count*CORR_UU_MEAN(i,k,j) + corr_uu )/(local_count+1.)
                CORR_VU_MEAN(i,k,j)  = (local_count*CORR_VU_MEAN(i,k,j) + corr_vu )/(local_count+1.)
                CORR_DUDT_MEAN(i,k,j)  = (local_count*CORR_DUDT_MEAN(i,k,j) + corr_dudt )/(local_count+1.)
            ENDIF
            IF (config_flags%output_vres_fluxes) THEN
                VX_MEAN(i,k,j) = (local_count*VX_MEAN(i,k,j) + vx*rhoxy )/(local_count+1.)/rhoxy_mean
                VY_MEAN(i,k,j) = (local_count*VY_MEAN(i,k,j) + vy*rho_d(i,k,j) )/(local_count+1.)/rho_mean
                VZ_MEAN(i,k,j) = (local_count*VZ_MEAN(i,k,j) + vz*rhoyz )/(local_count+1.)/rhoyz_mean
                RUV_TOT_MEAN(i,k,j)  = (local_count*RUV_TOT_MEAN(i,k,j) + ruv_tot(i,k,j) )/(local_count+1.)
                RVV_TOT_MEAN(i,k,j)  = (local_count*RVV_TOT_MEAN(i,k,j) + rvv_tot(i,k,j) )/(local_count+1.)
                WWV_TOT_MEAN(i,k,j)  = (local_count*WWV_TOT_MEAN(i,k,j) + wwv_tot(i,k,j) )/(local_count+1.)
                CORR_UV_MEAN(i,k,j)  = (local_count*CORR_UV_MEAN(i,k,j) + corr_uv )/(local_count+1.)
                CORR_VV_MEAN(i,k,j)  = (local_count*CORR_VV_MEAN(i,k,j) + corr_vv )/(local_count+1.)
                CORR_DVDT_MEAN(i,k,j)  = (local_count*CORR_DVDT_MEAN(i,k,j) + corr_dvdt )/(local_count+1.)
            ENDIF
            IF (config_flags%output_wres_fluxes) THEN
                WX_MEAN(i,k,j) = (local_count*WX_MEAN(i,k,j) + wx*rhoxz )/(local_count+1.)/rhoxz_mean
                WY_MEAN(i,k,j) = (local_count*WY_MEAN(i,k,j) + wy*rhoyz )/(local_count+1.)/rhoyz_mean
                WZ_MEAN(i,k,j) = (local_count*WZ_MEAN(i,k,j) + wz*rho_d(i,k,j) )/(local_count+1.)/rho_mean
                RUW_TOT_MEAN(i,k,j)  = (local_count*RUW_TOT_MEAN(i,k,j) + ruw_tot(i,k,j) )/(local_count+1.)
                RVW_TOT_MEAN(i,k,j)  = (local_count*RVW_TOT_MEAN(i,k,j) + rvw_tot(i,k,j) )/(local_count+1.)
                WWW_TOT_MEAN(i,k,j)  = (local_count*WWW_TOT_MEAN(i,k,j) + www_tot(i,k,j) )/(local_count+1.)
                CORR_UW_MEAN(i,k,j)  = (local_count*CORR_UW_MEAN(i,k,j) + corr_uw )/(local_count+1.)
                CORR_VW_MEAN(i,k,j)  = (local_count*CORR_VW_MEAN(i,k,j) + corr_vw )/(local_count+1.)
                CORR_DWDT_MEAN(i,k,j)  = (local_count*CORR_DWDT_MEAN(i,k,j) + corr_dwdt )/(local_count+1.)
            ENDIF
          ENDIF
            IF (config_flags%output_th_fluxes > 0) THEN
                T_TEND_RADLW_MEAN(i,k,j)  = (local_count*T_TEND_RADLW_MEAN(i,k,j) + t_tend_radlw(i,k,j) )/(local_count+1.)
                T_TEND_RADSW_MEAN(i,k,j)  = (local_count*T_TEND_RADSW_MEAN(i,k,j) + t_tend_radsw(i,k,j) )/(local_count+1.)
                T_TEND_MP_MEAN(i,k,j)  = (local_count*T_TEND_MP_MEAN(i,k,j) + t_tend_mp(i,k,j) )/(local_count+1.)
            ENDIF
            IF (config_flags%output_q_fluxes > 0) THEN
                Q_TEND_MP_MEAN(i,k,j)  = (local_count*Q_TEND_MP_MEAN(i,k,j) + qv_tend_mp(i,k,j) )/(local_count+1.)
            ENDIF
            !TODO: other mom sources (coriolis, curvature, small time step)
!~             IF (config_flags%output_u_fluxes > 0) THEN
!~             ENDIF
!~             IF (config_flags%output_v_fluxes > 0) THEN
!~             ENDIF
!~             IF (config_flags%output_w_fluxes > 0) THEN
!~             ENDIF
          !TODO: temperature and humidity tendencies from cumulus scheme
          IF (config_flags%output_sgs_fluxes) THEN
            IF (config_flags%output_thsgs_fluxes) THEN
                sgs_UTH_mean(i,k,j) = (local_count*sgs_UTH_mean(i,k,j) + sgs_UTH(i,k,j) ) / (local_count+1.)
                sgs_VTH_mean(i,k,j) = (local_count*sgs_VTH_mean(i,k,j) + sgs_VTH(i,k,j) ) / (local_count+1.)
                sgs_WTH_mean(i,k,j) = (local_count*sgs_WTH_mean(i,k,j) + sgs_WTH(i,k,j) ) / (local_count+1.)
            ENDIF
            IF (config_flags%output_qsgs_fluxes) THEN
                sgs_UQ_mean(i,k,j) = (local_count*sgs_UQ_mean(i,k,j) + sgs_UQ(i,k,j) ) / (local_count+1.)
                sgs_VQ_mean(i,k,j) = (local_count*sgs_VQ_mean(i,k,j) + sgs_VQ(i,k,j) ) / (local_count+1.)
                sgs_WQ_mean(i,k,j) = (local_count*sgs_WQ_mean(i,k,j) + sgs_WQ(i,k,j) ) / (local_count+1.)
            ENDIF
            IF (config_flags%output_usgs_fluxes) THEN
                sgs_UU_mean(i,k,j)  = (local_count*sgs_UU_mean(i,k,j)  + sgs_UU(i,k,j) )  / (local_count+1.)
                sgs_WU_mean(i,k,j)  = (local_count*sgs_WU_mean(i,k,j)  + sgs_WU(i,k,j) )  / (local_count+1.)
            ENDIF
            IF (config_flags%output_vsgs_fluxes) THEN
                sgs_VV_mean(i,k,j)  = (local_count*sgs_VV_mean(i,k,j)  + sgs_VV(i,k,j)  )  / (local_count+1.)
                sgs_WV_mean(i,k,j)  = (local_count*sgs_WV_mean(i,k,j)  + sgs_WV(i,k,j) )  / (local_count+1.)
            ENDIF
            IF (config_flags%output_usgs_fluxes .or. config_flags%output_vsgs_fluxes) THEN
                sgs_UV_mean(i,k,j)  = (local_count*sgs_UV_mean(i,k,j)  + sgs_UV(i,k,j) )  / (local_count+1.)
            ENDIF
            IF (config_flags%output_wsgs_fluxes) THEN
                sgs_UW_mean(i,k,j)  = (local_count*sgs_UW_mean(i,k,j)  + sgs_UW(i,k,j) )  / (local_count+1.)
                sgs_VW_mean(i,k,j)  = (local_count*sgs_VW_mean(i,k,j)  + sgs_VW(i,k,j) )  / (local_count+1.)
                sgs_WW_mean(i,k,j)  = (local_count*sgs_WW_mean(i,k,j)  + sgs_WW(i,k,j)  )  / (local_count+1.)
            ENDIF
          ENDIF

        end DO
      end DO
    end DO
    endif

    if (config_flags%do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = (local_count*avgflx_rum(i,k,j) + ru_m(i,k,j))/(local_count+1.)
             avgflx_rvm(i,k,j) = (local_count*avgflx_rvm(i,k,j) + rv_m(i,k,j))/(local_count+1.)
             avgflx_wwm(i,k,j) = (local_count*avgflx_wwm(i,k,j) + ww_m(i,k,j))/(local_count+1.)
          end DO
       end DO
    end DO
    endif

    if (do_cu .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) &
         & .and. present(cfu1) .and. present(cfd1) .and. present(dfu1) &
         & .and. present(efu1) .and. present(dfd1) .and. present(efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = (local_count*avgflx_cfu1(i,k,j) + &
                     & cfu1(i,k,j)) / (local_count+1.)
                avgflx_cfd1(i,k,j) = (local_count*avgflx_cfd1(i,k,j) + &
                     & cfd1(i,k,j)) / (local_count+1.)
                avgflx_dfu1(i,k,j) = (local_count*avgflx_dfu1(i,k,j) + &
                     & dfu1(i,k,j)) / (local_count+1.)
                avgflx_efu1(i,k,j) = (local_count*avgflx_efu1(i,k,j) + &
                     & efu1(i,k,j)) / (local_count+1.)
                avgflx_dfd1(i,k,j) = (local_count*avgflx_dfd1(i,k,j) + &
                     & dfd1(i,k,j)) / (local_count+1.)
                avgflx_efd1(i,k,j) = (local_count*avgflx_efd1(i,k,j) + &
                     & efd1(i,k,j)) / (local_count+1.)
             end DO
          end DO
       end DO
    end if

    return
  end subroutine upd_avgflx

  subroutine stagger_z(var_stag, var,                      &
                       cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                       i, j, k, kms, kme, kde              )

    REAL, INTENT(OUT) :: var_stag
    REAL, INTENT(IN) :: cf1, cf2, cf3, cfn, cfn1
    INTEGER, INTENT(IN) ::  i, j, k, kms, kme, kde
    REAL, DIMENSION(kms:kme) , INTENT(IN) :: var, fnm, fnp

    if ( k .eq. kde) then
      var_stag = cfn*var(kde-1)+cfn1*var(kde-2)
    elseif ( k .eq. 1) then
      var_stag = cf1*var(1)+cf2*var(2)+cf3*var(3)
    else
      var_stag = fnm(k)*var(k)+ fnp(k)*var(k-1)
    endif

  end subroutine stagger_z

  subroutine get_staggered_from_fluxes(varx, vary, varz,                &
                                       ruvar_tot, rvvar_tot, wwvar_tot, &
                                       ru, rv, ww, var,                 &
                                       cf1, cf2, cf3, cfn, cfn1, do_w,  &
                                       i, j, k,                         &
                                       ids, ide, jds, jde, kds, kde,    &
                                       ims, ime, jms, jme, kms, kme     )

    IMPLICIT NONE

    REAL, INTENT(OUT) :: varx, vary, varz
    REAL, INTENT(IN) :: ru, rv, ww, cf1, cf2, cf3, cfn, cfn1
    LOGICAL, INTENT(IN) :: do_w
    INTEGER, INTENT(IN) ::  i, j, k,                       &
                            ids, ide, jds, jde, kds, kde,  &
                            ims, ime, jms, jme, kms, kme
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
                        ruvar_tot, rvvar_tot, wwvar_tot, var

    WRITE ( wrf_err_message , FMT=* ) "i,j,k:",i,",",j,",",k
    if (ru .ne. 0) then
        varx = ruvar_tot(i,k,j)/ru
    else
        varx = var(i,k,j)
        if (k .lt. kde) then
            CALL wrf_debug ( 0 , ' Found zero ru when getting staggered variable from flux! Using var(i,k,j)...' )
            CALL wrf_debug ( 0 , wrf_err_message )
        endif
    endif
    if (rv .ne. 0) then
        vary = rvvar_tot(i,k,j)/rv
    else
        vary = var(i,k,j)
        if (k .lt. kde) then
            CALL wrf_debug ( 0 , ' Found zero rv when getting staggered variable from flux! Using var(i,k,j)...' )
            CALL wrf_debug ( 0 , wrf_err_message )
        endif
    endif

    if ((do_w) .and. (k .eq. kde)) THEN
      varz = 0
    elseif (k .eq. kde) then
      varz = cfn*var(i,kde-1,j)+cfn1*var(i,kde-2,j)
    elseif ((k .eq. 1) .and. (.not. do_w)) then
      varz = cf1*var(i,1,j)+cf2*var(i,2,j)+cf3*var(i,3,j)
    elseif (ww .ne. 0) then
      varz = wwvar_tot(i,k,j)/ww
    else
      varz = var(i,k,j)
      CALL wrf_debug ( 0 , ' Found zero ww when getting staggered variable from flux! Using var(i,k,j)...' )
      CALL wrf_debug ( 0 , wrf_err_message )
    endif
  end subroutine get_staggered_from_fluxes

end MODULE module_avgflx_em
