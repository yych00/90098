# Corrected-v2 HalfCheetah fixed-episode observations

This observation belongs to checkpoint SHA256 `0fb71a62b86d5135484da8982da6ddbe50f0cf2fde766ac8beee6b34d2cf9dbc`, corrected-v2 trainer hash `54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341`, training seed0 and evaluation seed10000. The fixed representative seed and episode follow the existing protocol; this is not a best-episode selection after comparing results.

The exported clip decoded all 1001 frames. The model viewed the six sampled frames at0,10,20,30,40 and50 seconds in representative_seed0_frames.png. This is a sampled-frame review, not continuous playback or a TensorBoard-browser check. The poses include pronounced pitching and a low body posture; forward displacement alone does not establish a smooth, stable gait.

Independent checkpoint replay reproduced all ten final evaluation returns exactly. The observed episode lasts 50.0 simulated seconds, returns 364.090150, and advances 39.964711m. Absolute wrapped torso pitch exceeds45 degrees in 34.07% of recorded states. This is a descriptive posture threshold, not a solved/termination rule. The checkpoint's worst final evaluation return is -441.048251; 1/10 episodes have negative return. These results do not establish uniformly reliable control.

Video SHA256 `bdc734c7e8cf2f82d64109a0b4d228aa11c4ce8967118ffe7a75b942fcc3a723`; sampled-image SHA256 `7a13a75bc2146c9841f1e116726fae827f058372c96d4686db663db5d471e034`. The render record, decode record and motion record bind the same checkpoint. Regenerate and inspect new media when reproducing with a different checkpoint; this archived observation applies only to the hashes above.

## 中文记录

本记录属于上述SHA绑定的corrected-v2模型，训练种子0、评估种子10000。已查看0、10、20、30、40、50秒六张抽样画面，看到明显俯仰和身体压低；没有声称连续看完整段视频或在浏览器验收TensorBoard。

实测回报364.090，模拟时长50.0秒，前进39.965米，躯干俯仰绝对值超过45度的状态比例为34.07%。该checkpoint的10个固定评估回合中有1个负回报回合；单段正回报与前进位移不足以证明步态平稳或控制始终可靠。复现时须重新生成并检查视频，不能把本记录换标签用于别的checkpoint。
