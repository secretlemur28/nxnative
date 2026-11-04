// Tessellated draws (NVN primitive type 14, patches) without a Metal tessellator: Metal 4 render encoders have no
// tessellation stage. Included after nvn_geometry.inc, whose vertex capture this reuses:
//   1. the vertex shader's capture entry writes each control point as a 144-word record (an attachment-less pass);
//   2. the hull shader runs as an ordinary kernel (dmsh_tess.py hull): one threadgroup per patch, its control points
//      read from the records through a control-point index list, its outputs in location slots, its factors in
//      SPIRV-Cross's MTLQuadTessellationFactorsHalf;
//   3. the domain shader runs as the vertex function of an ordinary draw (dmsh_tess.py domain) with the game's own
//      fragment shader and render state: each quad patch is an N x N grid of cells, two triangles each, N even
//      (fractional_even spacing has no odd levels). The hull's factors decide culling (an outer factor <= 0), not the
//      subdivision: adaptive levels are not implemented (DM_NVN_TESS_LEVEL picks N, default 8).
// Instances: each instance's control points are captured as their own records and its patches numbered after the
// previous instance's (the hull and domain shaders of this library use gl_PrimitiveID only to address their outputs).
// Draws this cannot replay (indirect, no translated hull / domain, a mode other than quads) stay skipped
// with a [RED] line, as before.
static id<MTL4ComputeCommandEncoder> dispatch_enc(void);
enum { TESS_CP_SLOTS = 33, TESS_PATCH_SLOTS = 32 };   // dmsh_tess.py CP_SLOTS / PATCH_SLOTS
static uint64_t g_tess_buf, g_tess_info;
static uint32_t g_tess_count;
static uint64_t g_tess_draws, g_tess_patches, g_tess_refused;
static int tess_stages(void) {   // DM_NVN_TESS_STAGES=1 / 2: diagnostic, stop after the vertex capture / the hull kernel (the draw is then skipped)
    static int n = 0; if (!n) { const char *e = getenv("DM_NVN_TESS_STAGES"); n = e ? atoi(e) : 3; if (n < 1 || n > 3) n = 3; } return n;
}
static int tess_level(void) {
    static int n = 0;
    if (!n) { const char *e = getenv("DM_NVN_TESS_LEVEL"); n = e ? atoi(e) : 8; n = n < 2 ? 2 : n > 64 ? 64 : n & ~1; }
    return n;
}
static void tess_refuse(NvProg *pr, const p_draw *dw, const p_draw_idx *di, const char *why) {
    g_tess_refused++;
    static int n; if (n++ < 8) MLOG("[RED] tessellated draw not replayed (vertex %s hull %s domain %s fragment %s; patch size %d, count %d, instances %d, indexed %d): %s",
        pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[3] ? pr->sh[3]->key : "-", pr->sh[4] ? pr->sh[4]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-",
        S.patch_size, dw ? dw->count : di->count, dw ? dw->instances : di->instances, di != NULL, why);
    PD("Z %llu draw skipped: tessellation not replayed (%s)", g_stat_pass + !g_enc, why);
}
static void tess_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    NvShader *vs = pr->sh[0], *hs = pr->sh[3], *ds = pr->sh[4];
    int count = dw ? dw->count : di->count, instances = dw ? dw->instances : di->instances, ps = S.patch_size;
    if (!vs || !vs->geo_fn) { tess_refuse(pr, dw, di, "no vertex capture entry"); return; }
    if (!hs || !hs->tess_ok || !ds || !ds->tess_ok) { tess_refuse(pr, dw, di, "hull or domain not rewritten for replay (dmsh_tess.py)"); return; }
    static const char *only; static int only_read;   // DM_NVN_TESS_ONLY=<hull key>,...: diagnostic, replay only those programs (to tell their effects apart)
    if (!only_read) { only = getenv("DM_NVN_TESS_ONLY"); only_read = 1; }
    if (only && !strstr(only, [[NSString stringWithUTF8String:hs->key] substringToIndex:MIN((NSUInteger)16, strlen(hs->key))].UTF8String)) { tess_refuse(pr, dw, di, "not in DM_NVN_TESS_ONLY"); return; }
    if (dw ? dw->ind || dw->mdi_cnt : di->ind || di->mdi_cnt) { tess_refuse(pr, dw, di, "indirect"); return; }
    if (instances < 1) { tess_refuse(pr, dw, di, "no instances"); return; }
    if (ps < 1 || ps > 32 || count < ps) { tess_refuse(pr, dw, di, "patch size"); return; }
    uint32_t np1 = (uint32_t)(count / ps), ncp = np1 * (uint32_t)ps, ni = (uint32_t)instances, np = np1 * ni, K = hs->tess_out_cp, N = (uint32_t)tess_level();
    uint32_t binst = (uint32_t)(dw ? dw->base_instance : di->base_instance);
    if (defer_has_active()) defer_fallback();
    lc_land(); rp_flush(); rp_end(); ubd_land();
    uint32_t *cpi = malloc((size_t)ncp * 4); if (!cpi) { tess_refuse(pr, dw, di, "no memory"); return; }
    uint32_t lo = UINT32_MAX, hi = 0;
    if (dw) { lo = (uint32_t)dw->first; hi = lo + ncp - 1; for (uint32_t i = 0; i < ncp; i++) cpi[i] = i; }
    else {
        int size = 1 << di->itype; uint64_t bytes = (uint64_t)ncp * size;
        if (di->itype < 0 || di->itype > 2) { free(cpi); tess_refuse(pr, dw, di, "index type"); return; }
        if (geo_index_wait(di->ibuf, bytes)) mdi_wait();
        if (!nvn_host_readable(di->ibuf, bytes)) { free(cpi); tess_refuse(pr, dw, di, "index buffer not readable"); return; }
        for (uint32_t i = 0; i < ncp; i++) { uint32_t n = 0; memcpy(&n, (void *)(uintptr_t)(di->ibuf + (uint64_t)i * size), size); cpi[i] = n;
            if (n < lo) lo = n; if (n > hi) hi = n; }
        if ((int64_t)lo + di->base_vertex < 0 || (int64_t)hi + di->base_vertex > UINT32_MAX) { free(cpi); tess_refuse(pr, dw, di, "base vertex"); return; }
        for (uint32_t i = 0; i < ncp; i++) cpi[i] -= lo;
        lo += di->base_vertex; hi += di->base_vertex;
    }
    uint64_t span = (uint64_t)hi - lo + 1, cap_bytes = span * ni * GEO_CAPTURE_WORDS * 4;
    uint64_t cp_bytes = (uint64_t)np * K * TESS_CP_SLOTS * 16, patch_off = cp_bytes, lev_off = patch_off + (uint64_t)np * TESS_PATCH_SLOTS * 16;
    uint64_t out_bytes = (lev_off + (uint64_t)np * 12 + 15) & ~15ull;
    if (!span || cap_bytes > (256u << 20) || out_bytes > (256u << 20) || (uint64_t)ncp * ni > (64u << 20) || !K || K > 32) { free(cpi); tess_refuse(pr, dw, di, "too large"); return; }
    id<MTLBuffer> b_cap, b_idx, b_out, b_par;
    size_t o_cap = arena_alloc(512 + cap_bytes, &b_cap), o_idx = arena_alloc((size_t)ncp * ni * 4, &b_idx), o_out = arena_alloc(out_bytes, &b_out), o_par = arena_alloc(64, &b_par);
    uint64_t g_cap = b_cap.gpuAddress + o_cap, g_idx = b_idx.gpuAddress + o_idx, g_out = b_out.gpuAddress + o_out, g_par = b_par.gpuAddress + o_par;
    { uint32_t *dst = (uint32_t *)((uint8_t *)b_idx.contents + o_idx);   // instance i's control point = its own record block (the capture writes (instance - base) * span + vertex - first)
      for (uint32_t i = 0; i < ni; i++) for (uint32_t c = 0; c < ncp; c++) dst[(size_t)i * ncp + c] = i * (uint32_t)span + cpi[c]; }
    free(cpi);
    uint32_t par[16] = {(uint32_t)ps, np, 0, 0, N, K, (uint32_t)patch_off, (uint32_t)ds->tess_cw, (uint32_t)lev_off, 0, 0, 0};
    memcpy((uint8_t *)b_par.contents + o_par, par, sizeof par);   // [0..1]: spvIndirectParams; [4..11]: the domain's dm_tinfo[0..1]
    nvn_geo_capture gc = {g_cap + 512, 0, {lo, (uint32_t)span, binst, ni}, GEO_CAPTURE_WORDS, {0}};
    for (int k = 0; k < GEO_CAPTURE_WORDS; k++) gc.slot[k] = (int16_t)k;
    memcpy((uint8_t *)b_cap.contents + o_cap, &gc, sizeof gc);
    // 1. control points through the vertex shader's capture entry
    id<MTLRenderPipelineState> cap = geo_capture_pso(vs); if (!cap) { tess_refuse(pr, dw, di, "vertex capture pipeline"); return; }
    p_draw range = {.first = (int32_t)lo, .count = (int32_t)span, .instances = (int32_t)ni, .base_instance = (int32_t)binst};
    vbuf_prepare(&vs->rf, &range, NULL);
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    g_draw_binst = binst;
    MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new]; rd.renderTargetWidth = rd.renderTargetHeight = rd.defaultRasterSampleCount = 1;
    id<MTL4RenderCommandEncoder> re = enc_r_begin(rd); enc_r_pso(re, cap); bind_vbufs(re);
    vconv_bind(re, vs, 1, &range, NULL); args_dirty_all(); encode_args(re, vs, 0, 1); enc_r_vbuf(re, 13, g_cap);
    enc_r_draw_instanced(re, MTLPrimitiveTypePoint, lo, span, ni, binst); enc_r_end(re);
    if (tess_stages() < 2) { g_tess_draws++; return; }
    // 2. the hull shader, one threadgroup per patch
    if (!hs->geo_pso && !hs->geo_pso_failed) {
        NSError *e = nil; hs->geo_pso = [g_dev newComputePipelineStateWithFunction:hs->fn error:&e];
        if (!hs->geo_pso) { hs->geo_pso_failed = 1; MLOG("[RED] hull compute %s: %s", hs->key, e.localizedDescription.UTF8String); }
    }
    uint32_t threads = (uint32_t)(ps > (int)K ? ps : (int)K);
    if (!hs->geo_pso || hs->geo_pso.maxTotalThreadsPerThreadgroup < threads) { tess_refuse(pr, dw, di, "hull pipeline"); return; }
    // the shared compute / copy encoder, as dispatch(): a texture upload or upgrade encode_args needs lands in this same open encoder
    id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)dispatch_enc();
    enc_c_pso(ce, hs->geo_pso); args_dirty_all(); encode_args(ce, hs, 3, 0);
    enc_c_args(ce, g_idx, 24); enc_c_args(ce, g_cap + 512, 25); enc_c_args(ce, g_out + lev_off, 26); enc_c_args(ce, g_out + patch_off, 27); enc_c_args(ce, g_out, 28); enc_c_args(ce, g_par, 29);
    [ce setThreadgroupMemoryLength:((NSUInteger)ps * hs->tess_in_bytes + 15) & ~(NSUInteger)15 atIndex:0];
    enc_c_dispatch(ce, MTLSizeMake(np, 1, 1), MTLSizeMake(threads, 1, 1));
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }   // the draw's render pass follows; its full barrier orders it after the kernel
    swrite_note(pr); g_wr_epoch++; g_mdx_epoch++;
    if (tess_stages() < 3) { g_tess_draws++; return; }
    // 3. the domain shader as the vertex stage of an ordinary draw
    NvProg *raster = pr->tess_raster;
    if (!raster || raster->sh[0] != ds || raster->sh[1] != pr->sh[1]) { raster = [NvProg new]; raster->sh[0] = ds; raster->sh[1] = pr->sh[1]; pr->tess_raster = raster; }
    g_geo_program = raster; g_tess_buf = g_out; g_tess_info = g_par + 16; g_tess_count = np * 6 * N * N;
    args_dirty_all();
    p_draw synthetic = {.prim = 4, .first = 0, .count = (int32_t)g_tess_count, .instances = 1, .base_instance = (int32_t)binst};
    draw(&synthetic, NULL);
    g_geo_program = nil; g_tess_buf = g_tess_info = 0; g_tess_count = 0; args_dirty_all();
    g_tess_draws++; g_tess_patches += np;
    static int once; if (once++ < 4) MLOG("tessellation: vertex %s hull %s domain %s, %u patches (%u per instance x %u) of %d control points (%u out), grid %u, %s", vs->key, hs->key, ds->key, np, np1, ni, ps, K, N, di ? "indexed" : "arrays");
}
