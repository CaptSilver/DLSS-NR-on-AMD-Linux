/* A mod session restart on the same flag buffer reuses the token and restarts
 * frame numbers at 1. Production bridge with CPU-only stand-ins for HIP. */
#include "../native/hip_bridge.c"
#include <assert.h>

static int refs;
static uint64_t token;
static int retain(uint64_t owner) { assert(owner==77); ++refs; return 0; }
static void release(uint64_t owner) { assert(owner==77); --refs; }
static int fail_probe_launch, publish_in_sync;
static struct nr_buffer snapshot[2];
static uint32_t count;
/* Runs inside the kind-3 probe: a frame published here makes the resource busy
 * by the time the probe finishes. */
static int native_sync(void *stream) {
    (void)stream;
    if (publish_in_sync) {
        publish_in_sync=0;
        assert(!nr_snapshot(1,snapshot,2,&count));
        assert(!nr_publish_input(token,2,snapshot,count));
        snapshot[0].release(snapshot[0].owner);
    }
    return 0;
}
static int device_sync(void) { return 0; }
static int native_launch(const void *f,dim3_t grid,dim3_t block,void **args,u64 shared,void *stream) {
    (void)f;(void)grid;(void)block;(void)args;(void)shared;(void)stream;
    if (fail_probe_launch) { fail_probe_launch=0; return 1; }
    return 0;
}
static uint32_t flags[16];
static void *flag_ptr=flags,*abort_ptr=(void *)0x3000,*no_abort;

static int send(const void *fn,void **args) {
    return stub_launch(fn,(dim3_t){1,1,1},(dim3_t){32,1,1},args,0,NULL);
}
static void probe_kind(unsigned kind) {
    uint32_t zero=0,magic=UINT32_C(0x40000000),one=1,limit=1500000;
    void *wait1[]={&flag_ptr,&zero,&one},*set[]={&flag_ptr,&zero,&no_abort};
    void *wait3[]={&flag_ptr,&magic,&limit};
    assert(!send(kind==2?g_flag_set:g_flag_wait,kind==1?wait1:kind==2?set:wait3));
}
/* The startup probe sequence exactly as the mod issues it. */
static void probe_sequence(void) { probe_kind(1); probe_kind(2); probe_kind(3); }

/* The HIP side of a frame whose input is already published. */
static void run_frame(uint32_t frame) {
    uint32_t limit=20000000;
    void *wait[]={&flag_ptr,&frame,&limit},*set[]={&flag_ptr,&frame,&abort_ptr};
    void *ordinary[]={&flag_ptr};
    assert(!send(g_flag_wait,wait) && nr_job.active);
    assert(!send(g_k_import,ordinary));
    assert(!send(g_k_export,ordinary));
    assert(!send(g_flag_set,set) && !nr_job.active);
    assert(!nr_wait_output(token,frame,0));
    assert(!nr_consumer_submitted(token,frame));
}
static void complete_frame(uint32_t frame) {
    assert(!nr_snapshot(1,snapshot,2,&count) && count==1);
    assert(!nr_publish_input(token,frame,snapshot,count));
    snapshot[0].release(snapshot[0].owner);
    run_frame(frame);
}

int main(int argc,char **argv) {
    assert(argc==2);int scenario=atoi(argv[1]);
    g_hip_ok=1;real.p_hipLaunchKernel=native_launch;
    real.p_hipStreamSynchronize=native_sync;real.p_hipDeviceSynchronize=device_sync;
    g_flag_wait=(void *)1;g_flag_set=(void *)2;g_k_import=(void *)3;g_k_export=(void *)4;
    assert(!nr_register_resource(1,2,3,4,sizeof(flags),77,retain,release,&token));
    nr_associate_import(2,(void *)5);nr_associate_mapping((void *)5,flags,0,sizeof(flags));
    assert(!nr_snapshot(1,snapshot,2,&count) && count==1);
    assert(!nr_claim_queue(snapshot,1,9));
    snapshot[0].release(snapshot[0].owner);assert(refs==0);

    probe_sequence();
    complete_frame(1);
    uint64_t again;
    if(scenario==0) {
        /* Re-created staging: same flag buffer, same token, probe, frame 1 again. */
        assert(!nr_register_resource(1,6,3,4,sizeof(flags),77,retain,release,&again) && again==token);
        probe_sequence();
        complete_frame(1);
        complete_frame(2);
        /* Replay inside the new session is still refused. */
        assert(!nr_snapshot(1,snapshot,2,&count));
        assert(nr_publish_input(token,2,snapshot,count)!=0);
        snapshot[0].release(snapshot[0].owner);
    } else if(scenario==1) {
        /* No probe: a replayed frame number is refused. */
        assert(!nr_snapshot(1,snapshot,2,&count));
        assert(nr_publish_input(token,1,snapshot,count)==1);
        snapshot[0].release(snapshot[0].owner);
    } else if(scenario==2) {
        /* Incomplete probe (kinds 1 and 2 only) is not a new session. */
        probe_kind(1);probe_kind(2);
        assert(!nr_snapshot(1,snapshot,2,&count));
        assert(nr_publish_input(token,1,snapshot,count)==1);
        snapshot[0].release(snapshot[0].owner);
    } else if(scenario==3) {
        /* A failed kind-3 probe does not start a new session. */
        probe_kind(1);probe_kind(2);
        fail_probe_launch=1;
        uint32_t magic=UINT32_C(0x40000000),limit=1500000;
        void *wait3[]={&flag_ptr,&magic,&limit};
        send(g_flag_wait,wait3);
        assert(!nr_snapshot(1,snapshot,2,&count));
        assert(nr_publish_input(token,1,snapshot,count)==1);
        snapshot[0].release(snapshot[0].owner);
    } else if(scenario==4) {
        /* Busy when the probe finishes: the reset is skipped and the live
         * frame keeps its pins. */
        probe_kind(1);probe_kind(2);
        publish_in_sync=1;
        probe_kind(3);
        struct nr_frame_state *s=nr_frame(nr_find_token(token),2,0);
        assert(s && s->input_ready && s->pins);
        run_frame(2);
    } else return 2;
    assert(refs==0);
    printf("scenario %d passed\n",scenario);return 0;
}
