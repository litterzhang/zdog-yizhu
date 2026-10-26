import { brand } from './brand'
import type { DialogOptions } from '../services/dialog'

const information = (title: string, content: string): DialogOptions => ({
  title, content, showCancel: false,
})

const confirmation = (
  title: string, content: string, confirmText: string, tone: DialogOptions['tone'] = 'normal',
): DialogOptions => ({ title, content, confirmText, tone })

export const dialogCopy = {
  loginPrivacy: information('登录与隐私说明',
    '登录信息用于验证身份，本机不保存密码，也不获取手机号、头像或定位。\n\n退出仅清除本机登录状态，不会注销账号或退出其他设备。'),
  usage: information('使用说明',
    '选一张植物照片，确认物种，再把它记入株记。\n\n还不知道名字也可以记录，以后再慢慢补充。后续照片可以继续添加到同一株植物。'),
  privacy: information('数据与隐私',
    '照片仅在你主动识别或保存观察时上传，与足迹、株记一起按账号私有保存。\n\n不获取定位，不保存登录密码。退出登录不会删除记录，删除记录不会注销账号。\n\n需要注销账号，请联系账号服务维护者。'),
  about: information(`关于${brand.name}`,
    `${brand.description}\n\n留下一份自己的自然手账，记住每一次相遇。\n\n服务可能暂时离线，且尚无自动异地备份，请勿把这里作为重要记录的唯一保存处。`),
  photoCredits: information('照片来源',
    '内置银杏照片：Tangopaso / Wikimedia Commons，公有领域。\n作品：Ginkgo leaves (Vaires) 1 / 5。\n\n图片已缩放、压缩并去除元数据。'),
  browserPermissions: information('检查浏览器权限',
    '请在浏览器的站点设置中检查相机或照片权限，再返回重试。\n\n电脑浏览器可能只支持从文件中选图。'),
  wechatPermissions: information('暂时无法打开设置',
    '请在微信中检查一株的相机与相册权限，再返回重试。'),
  recognition: information('识别说明',
    `结果可能有误，请结合植物特征确认物种；不确定时可以暂不确认。\n\n${brand.recognitionSafetyNotice}`),
  signOut: confirmation('退出登录？',
    '将清除本机登录状态、未保存的草稿和缓存。已保存的记录与照片保留，重新登录即可找回。\n\n不会注销账号，也不会退出其他设备。', '确认退出'),
  clearData: confirmation('删除我的全部数据？',
    '将删除全部足迹、株记、观察和已上传照片，并取消进行中的识别。\n\n此操作无法撤销，不会删除手机相册原图或注销账号。', '确认清空', 'danger'),
  saveObservation: confirmation('保存这次观察？', brand.uploadNotice, '上传保存'),
  removeObservation: confirmation('删除这次观察？',
    '将删除这次观察，保留株记及其他观察。\n\n对应足迹仍关联此株记；封面或其他记录使用的照片会保留。此操作无法撤销。', '确认删除', 'danger'),
  removeRecognition: confirmation('删除这条足迹？',
    '将删除这条足迹，并取消尚未完成的识别。已建立的株记和观察不受影响。\n\n其他记录仍使用的照片会保留，其余照片将删除。此操作无法撤销。', '确认删除', 'danger'),
  linkPlant: (name: string) => confirmation('确定是同一株吗？',
    `将这张照片记入「${name}」。请确认它们是同一株植物，而不只是同一种。\n\n一条足迹只能关联一份株记，已确认的物种及之后的物种修改都会同步到该株记。`, '是同一株'),
  removePlant: (name: string) => confirmation(`删除「${name}」？`,
    '将删除这份株记及全部观察，足迹保留并解除关联。\n\n其他记录仍使用的照片会保留，其余照片将删除。此操作无法撤销。', '确认删除', 'danger'),
}
